from PIL import Image
import numpy as np
import tensorflow as tf
from dataset import CifarGenerator
from functools import partial


class CifarGeneratorTTA(CifarGenerator):
    def __init__(self, *args, n_tta=3, multiprocess=False, **kwargs):
        """
        :param data: (array) - image data (n, h, w, c). Ex) x_train: (50000, 32, 32, 3)
        :param label: (array) - label data (n, )
        :param augment: (bool)
        :param model_type: (str) - Name of the model. Pre-processing will be applied according to the model described in TensorFlow documents.
        :param image_size: (tuple) - Target image size. CifarGenerator resize the image as image_size.
        :param ignore_model_preprocess: (bool) - True: Ignores pre-processing methods described in TensorFlow. Instead, it normalizes data dividing by 255.0.
        """
        super(CifarGeneratorTTA, self).__init__(*args, **kwargs)
        self.n_tta = n_tta
        assert self.augment_func is not None, "Augmentation function must be defined!"

        self.multiprocess = multiprocess

    @staticmethod
    def _tta_generator(imgs=None, labels=None, augment_pipeline=None, n_tta=3,
                       data_format="channels_first"):

        for img, label in zip (imgs, labels):

            tta_imgs = [augment_pipeline(img) for _ in range(n_tta)]

            tta_imgs = np.concatenate(tta_imgs, axis=-1)
            tta_imgs = np.swapaxes(tta_imgs.T, 1, 2) if data_format == "channels_first" else tta_imgs

            yield tta_imgs, label

    @staticmethod
    def augment_pipeline(img, augment_func=None, preprocess_func=None, augment_in_dtype="pil",
                         image_size=(32, 32), dtype=np.float32):
        ori_img = img
        img = CifarGeneratorTTA.resize_image(ori_img, image_size=image_size)
        img = CifarGeneratorTTA.apply_augment(img, augment_func=augment_func, augment_in_dtype=augment_in_dtype)
        img = preprocess_func(img, dtype=dtype)

        return img

    @staticmethod
    def apply_augment(img, augment_func=None, augment_in_dtype="pil"):
        if augment_func is not None:
            if type(img) != np.ndarray and augment_in_dtype == 'numpy':
                img = np.array(img, dtype=np.uint8)
            if type(img) == np.ndarray and augment_in_dtype == 'pil':
                img = Image.fromarray(img)

            img = augment_func(img)

        return img

    @staticmethod
    def resize_image(img, image_size=(32, 32)):
        if image_size == (32, 32):
            return img

        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        img = img.resize(image_size)

        return img

    def get_tf_dataset(self, batch_size):
        pipeline = partial(CifarGeneratorTTA.augment_pipeline,
                           augment_func=self.augment_func,
                           augment_in_dtype = self.augment_in_dtype,
                           preprocess_func=self.preprocess_func,
                           image_size=self.image_size, dtype=self.dtype
                           )
        generator = partial(CifarGeneratorTTA._tta_generator,
                            imgs=self.data, labels=self.label,
                            augment_pipeline=pipeline,
                            n_tta=self.n_tta,
                            data_format=self.data_format)

        img_shape = tf.TensorShape([self.image_size[1], self.image_size[0],
                                    3 if self.augment_in_dtype == "tensor" else 3 * self.n_tta])

        dataset = tf.data.Dataset.from_generator(generator,
                                                 (tf.as_dtype(self.dtype), tf.int32),
                                                 (img_shape, tf.TensorShape([])))

        dataset = dataset.shuffle(self.shuffle_size, reshuffle_each_iteration=True) if self.shuffle else dataset

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

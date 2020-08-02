from PIL import Image
import numpy as np
import tensorflow as tf
from dataset import preprocessing


class CifarGenerator:
    def __init__(self, data, label, augment_func=None, augment_in_dtype="numpy", preprocess_func=preprocessing.preprocess_default,
                 image_size=(32, 32), dtype=np.float32, seed=7777, data_format="channels_last", shuffle=False, shuffle_size=64):
        """
        :param data: (array) - image data (n, h, w, c). Ex) x_train: (50000, 32, 32, 3)
        :param label: (array) - label data (n, )
        :param augment: (bool)
        :param model_type: (str) - Name of the model. Pre-processing will be applied according to the model described in TensorFlow documents.
        :param image_size: (tuple) - Target image size. CifarGenerator resize the image as image_size.
        :param ignore_model_preprocess: (bool) - True: Ignores pre-processing methods described in TensorFlow. Instead, it normalizes data dividing by 255.0.
        """
        assert augment_in_dtype in ['numpy', 'pil', "tensor"]

        self.data = data
        self.label = label
        self.augment_func = augment_func
        self.preprocess_func = preprocess_func
        self.dtype = dtype
        self.seed = seed
        self.augment_in_dtype = augment_in_dtype
        self.image_size = image_size
        self.data_format = data_format
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size

    def apply_augment(self, img):
        if self.augment_func is not None:
            if type(img) != np.ndarray and self.augment_in_dtype == 'numpy':
                img = np.array(img, dtype=np.uint8)
            if type(img) == np.ndarray and self.augment_in_dtype == 'pil':
                img = Image.fromarray(img)

            img = self.augment_func(img)

        return img

    def resize_image(self, img):
        if self.image_size == (32, 32):
            return img

        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        img = img.resize(self.image_size)

        return img

    def __call__(self):
        for img, l in zip(self.data, self.label):
            img = self.resize_image(img)
            img = self.apply_augment(img)
            img = self.preprocess_func(img)
            img = np.swapaxes(img.T, 1, 2) if self.data_format == "channels_first" else img

            yield img, l

    def get_tf_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_generator(self,
                                                 (tf.as_dtype(self.dtype), tf.int32),
                                                 (tf.TensorShape([self.image_size[1], self.image_size[0], 3]),
                                                  tf.TensorShape([])))

        dataset = dataset.shuffle(self.shuffle_size, reshuffle_each_iteration=True) if self.shuffle else dataset

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

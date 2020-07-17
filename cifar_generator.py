from autoaugment import CIFAR10Policy
from PIL import Image
import numpy as np
import tensorflow as tf


class CifarGenerator:
    def __init__(self, data, label, augment=False, model_type=None, image_size=(32, 32), ignore_model_preprocess=False):
        """
        :param data: (array) - image data (n, h, w, c). Ex) x_train: (50000, 32, 32, 3)
        :param label: (array) - label data (n, )
        :param augment: (bool)
        :param model_type: (str) - Name of the model. Pre-processing will be applied according to the model described in TensorFlow documents.
        :param image_size: (tuple) - Target image size. CifarGenerator resize the image as image_size.
        :param ignore_model_preprocess: (bool) - True: Ignores pre-processing methods described in TensorFlow. Instead, it normalizes data dividing by 255.0.
        """
        self.data = data
        self.label = label
        self.augment = augment
        self.model_type = model_type
        self.image_size = image_size
        self.preprocess_func = lambda x: np.array(x, dtype=np.float32) / 255.0

        """
        Image Pre-processing described as in TensorFlow Documents
        https://www.tensorflow.org/api_docs/python/tf/keras/applications
        >>> tf.keras.applications.imagenet_utils.preprocess_input
        >>> tf.keras.applications.imagenet_utils._preprocess_numpy_input
        """
        if model_type is not None and not self.model_type.endswith("custom") and not ignore_model_preprocess:
            if self.model_type.startswith("mobilenet") or \
                    (self.model_type.startswith("resnet") and self.model_type.endswith("v2")) or \
                    self.model_type.startswith("inception") or \
                    self.model_type.startswith("xception"):
                # -1 to 1
                self.preprocess_func = lambda x: (np.array(x, dtype=np.float32) / 127.5)-1
            elif self.model_type.startswith("resnet") or self.model_type.startswith("VGG"):
                # RGB -> BGR then zero-centered(ImageNet) without scaling
                self.preprocess_func = lambda x: (np.array(x, dtype=np.float32)[..., ::-1] - np.array([103.939, 116.779, 123.67]))
            elif self.model_type.startswith("effnet"):
                self.preprocess_func = lambda x: ((np.array(x, dtype=np.float32) / 255.0) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    def __call__(self):
        policy = CIFAR10Policy() if self.augment else None

        for d, l in zip(self.data, self.label):
            d = Image.fromarray(d).resize(self.image_size)
            if policy:
                d = policy(d)

            yield self.preprocess_func(d), l

    def get_tf_dataset(self, batch_size, shuffle=False, reshuffle=True, shuffle_size=64):
        dataset = tf.data.Dataset.from_generator(self,
                                                 (tf.float32, tf.int32),
                                                 (tf.TensorShape([self.image_size[0], self.image_size[1], 3]),
                                                  tf.TensorShape([]))).batch(batch_size)

        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=reshuffle) if shuffle else dataset
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

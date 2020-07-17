from autoaugment import CIFAR10Policy
from PIL import Image
import numpy as np
import tensorflow as tf


class Cifar100Generator:
    def __init__(self, data, label, augment=False, model_type=None, image_size=(224, 224)):
        self.data = data
        self.label = label
        self.augment = augment
        self.model_type = model_type
        self.image_size = image_size
        self.preprocess_func = lambda x: np.array(x, dtype=np.float32) / 255.0

        if model_type is not None and not self.model_type.endswith("custom"):
            if self.model_type.startswith("mobilenet") or (self.model_type.startswith("resnet") and self.model_type.endswith("v2")):
                # -1 to 1
                self.preprocess_func = lambda x: (np.array(x, dtype=np.float32) / 127.5)-1
            elif self.model_type.startswith("resnet"):
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

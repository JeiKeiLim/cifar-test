import tensorflow as tf


class Hswish:
    def __init__(self, name=None, dtype=tf.float32):
        self.name = name
        self.dtype = dtype

    def __call__(self, x):
        return x * tf.nn.relu6(x + 3, name=self.name) / 6


class Swish:
    def __init__(self, name=None, dtype=tf.float32):
        self.name = name
        self.dtype = dtype

    def __call__(self, x):
        with tf.name_scope(self.name):
            return tf.nn.swish(x)

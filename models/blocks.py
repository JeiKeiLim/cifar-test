import tensorflow as tf


class BottleNeckBlock:
    def __init__(self, n_filter, kernel_size, downsample=False, downsample_strides=(2, 2), padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="bottle_resblock", out_filter_multiplier=4, dtype=tf.float32):
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.padding = padding
        self.activation = activation
        self.name = name
        self.use_bias = use_bias
        self.downsample_strides = downsample_strides
        self.out_filter_multiplier = out_filter_multiplier
        self.dtype = dtype

    def __call__(self, layer):
        strides = self.downsample_strides if self.downsample else (1, 1)

        x = ConvBN(self.n_filter, (1, 1), strides=(1, 1), padding=self.padding, use_bias=self.use_bias,
                   name=f"{self.name}_conv_bn_front", dtype=self.dtype)(layer)

        x = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size, strides=strides, use_bias=self.use_bias,
                                   activation=None, padding=self.padding, dtype=self.dtype,
                                   name="{}_{}x{}conv_0".format(self.name, self.kernel_size[0], self.kernel_size[1]))(x)
        if layer.shape[-1] != self.out_filter_multiplier:
            layer = tf.keras.layers.Conv2D(self.n_filter*self.out_filter_multiplier, (1, 1), strides=strides, use_bias=self.use_bias,
                                           activation=None, padding=self.padding, dtype=self.dtype,
                                           name=f"{self.name}_1x1conv_init")(layer)

        x = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_0", dtype=self.dtype)(x)
        x = self.activation(name=f"{self.name}_activation_0", dtype=self.dtype)(x)
        x = tf.keras.layers.Conv2D(self.n_filter*self.out_filter_multiplier, (1, 1),
                                   strides=(1, 1), use_bias=self.use_bias, dtype=self.dtype,
                                   name=f"{self.name}_1x1conv_1")(x)

        x_layer = tf.keras.layers.Add(name=f"{self.name}_residual", dtype=self.dtype)([x, layer])
        x_layer = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_1", dtype=self.dtype,)(x_layer)
        x_layer = self.activation(name=f"{self.name}_activation_1", dtype=self.dtype,)(x_layer)

        return x_layer


class ResNetBlock:
    def __init__(self, n_filter, kernel_size, downsample=False, padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="resblock", dtype=tf.float32):
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.padding = padding
        self.activation = activation
        self.name = name
        self.use_bias = use_bias
        self.dtype=dtype

    def __call__(self, layer):
        strides = (2, 2) if self.downsample else (1, 1)

        x = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size, strides=strides, padding=self.padding,
                                   activation=None, use_bias=self.use_bias, dtype=self.dtype,
                                   name="{}_{}x{}conv_0".format(self.name, self.kernel_size[0], self.kernel_size[1]))(layer)
        if self.downsample:
            layer = tf.keras.layers.Conv2D(self.n_filter, kernel_size=(1, 1), strides=(2, 2),
                                           padding=self.padding, activation=None, dtype=self.dtype,
                                           name="{}_{}x{}conv_init".format(self.name, self.kernel_size[0], self.kernel_size[1]))(layer)

        x = tf.keras.layers.BatchNormalization(name="{}_bn_0".format(self.name), dtype=self.dtype)(x)
        x = self.activation(name="{}_activation_0".format(self.name), dtype=self.dtype)(x)
        x = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size, strides=(1, 1),
                                   use_bias=self.use_bias, activation=None, padding=self.padding, dtype=self.dtype,
                                   name="{}_{}x{}conv_1".format(self.name, self.kernel_size[0], self.kernel_size[1]))(x)

        x_layer = tf.keras.layers.Add(name=f"{self.name}_residual", dtype=self.dtype)([x, layer])
        x_layer = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_1", dtype=self.dtype)(x_layer)
        x_layer = self.activation(name=f"{self.name}_activation_1", dtype=self.dtype)(x_layer)

        return x_layer


class ConvBN:
    def __init__(self, n_filter, kernel_size, strides=(1, 1), padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="conv_bn", dtype=tf.float32):
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.name = name
        self.dtype = dtype

    def __call__(self, layer):
        layer = tf.keras.layers.Conv2D(self.n_filter, self.kernel_size,
                                       strides=self.strides, padding=self.padding,
                                       activation=None, use_bias=self.use_bias, dtype=self.dtype,
                                       name="{}_{}x{}conv".format(self.name, self.kernel_size[0], self.kernel_size[1]))(layer)
        layer = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn", dtype=self.dtype)(layer)
        layer = self.activation(name=f"{self.name}_activation", dtype=self.dtype)(layer)

        return layer

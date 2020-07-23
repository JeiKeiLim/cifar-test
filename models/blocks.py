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


class DenseBottleNeck:
    def __init__(self, expansion=4, growth_rate=12, kernel_size=(3, 3),
                 p_drop=0.0, padding='SAME', use_bias=False, Activation=tf.keras.layers.ReLU,
                 Conv2D=tf.keras.layers.Conv2D,
                 name="dense_bottle", dtype=tf.float32):

        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{name}_bn1", dtype=dtype)
        self.conv1 = tf.keras.layers.Conv2D(expansion * growth_rate, (1, 1), strides=(1, 1), dtype=dtype,
                                            padding=padding, use_bias=use_bias,
                                            name=f"{name}_conv1")

        self.bn2 = tf.keras.layers.BatchNormalization(name=f"{name}_bn2", dtype=dtype)
        self.conv2 = Conv2D(growth_rate, kernel_size, strides=(1, 1), dtype=dtype,
                                            padding=padding, use_bias=use_bias,
                                            name=f"{name}_conv2")
        self.dropout = tf.keras.layers.Dropout(p_drop, name=f"{name}_dropout") if p_drop > 0.0 else None
        self.activation = Activation(name=f"{name}_act", dtype=dtype)
        self.concat = tf.keras.layers.Concatenate(name=f"{name}_concat")

    def __call__(self, layer):
        x = self.bn1(layer)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.dropout:
            x = self.dropout(x)

        x = self.concat([layer, x])

        return x


class DenseBasic:
    def __init__(self, growth_rate=12, kernel_size=(3, 3), p_drop=0.0, padding='SAME',
                 use_bias=False, Activation=tf.keras.layers.ReLU, Conv2D=tf.keras.layers.Conv2D,
                 name="dense_basic", dtype=tf.float32, **kwargs):

        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{name}_bn1", dtype=dtype)
        self.conv1 = Conv2D(growth_rate, kernel_size, strides=(1, 1), dtype=dtype,
                                            padding=padding, use_bias=use_bias,
                                            name=f"{name}_conv1")
        self.activation = Activation(name=f"{name}_act", dtype=dtype)
        self.dropout = tf.keras.layers.Dropout(p_drop, name=f"{name}_dropout") if p_drop > 0.0 else None
        self.concat = tf.keras.layers.Concatenate(name=f"{name}_concat")

    def __call__(self, layer):
        x = self.bn1(layer)
        x = self.activation(x)
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.concat([layer, x])

        return x


class DenseReduce:
    def __init__(self, n_filter, downsample_kernel_size=(3, 3), downsample_strides=(2, 2), padding='SAME', use_bias=False,
                 Activation=tf.keras.layers.ReLU, name="dense_reduce", dtype=tf.float32):
        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{name}_bn1", dtype=dtype)
        self.conv1 = tf.keras.layers.Conv2D(n_filter, (1, 1), strides=(1, 1), dtype=dtype,
                                            padding=padding, use_bias=use_bias,
                                            name=f"{name}_conv1")
        self.activation = Activation(name=f"{name}_act", dtype=dtype)
        self.reduce = tf.keras.layers.AveragePooling2D(downsample_kernel_size, downsample_strides, padding=padding,
                                                       name=f"{name}_reduce")

    def __call__(self, layer):
        x = self.bn1(layer)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.reduce(x)

        return x


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


class SeparableConvBN:
    def __init__(self, n_filter, kernel_size, strides=(1, 1), padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="sep_conv_bn", dtype=tf.float32):
        self.conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                                                     activation=None, name=f"{name}_conv1", dtype=dtype)
        self.bn1 = tf.keras.layers.BatchNormalization(name=f"{name}_bn1", dtype=dtype)
        self.activation = activation(name=f"{name}_act", dtype=dtype)

        self.conv2 = tf.keras.layers.Conv2D(n_filter, (1, 1), strides=(1, 1), padding=padding, use_bias=use_bias,
                                            activation=None, name=f"{name}_conv2", dtype=dtype)
        self.bn2 = tf.keras.layers.BatchNormalization(name=f"{name}_bn2", dtype=dtype)

    def __call__(self, layer):
        x = self.conv1(layer)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        return x


class ConvBN:
    def __init__(self, n_filter, kernel_size, strides=(1, 1), padding='SAME', use_bias=True,
                 activation=tf.keras.layers.ReLU, name="conv_bn", Conv=tf.keras.layers.Conv2D,
                 dtype=tf.float32):
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.name = name
        self.dtype = dtype
        self.Conv = Conv

    def __call__(self, layer):
        layer = self.Conv(self.n_filter, self.kernel_size,
                                       strides=self.strides, padding=self.padding,
                                       activation=None, use_bias=self.use_bias, dtype=self.dtype,
                                       name="{}_{}x{}conv".format(self.name, self.kernel_size[0], self.kernel_size[1]))(layer)
        layer = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn", dtype=self.dtype)(layer)
        layer = self.activation(name=f"{self.name}_activation", dtype=self.dtype)(layer)

        return layer

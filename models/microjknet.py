import tensorflow as tf
from models import DenseBottleNeck, DenseReduce, ConvBN, SeparableConvBN


class MicroJKNet:
    def __init__(self, input_shape=(None, None, 3), growth_rate=12, depth=3, in_depth=3,
                 Block=DenseBottleNeck, Conv=tf.keras.layers.Conv2D, Activation=tf.keras.layers.ReLU,
                 expansion=4, n_classes=0, p_drop=0.0,
                 compression_rate=2, reduce_kernel_size=(3, 3),
                 float16=False, float16_dtype='mixed_float16'):
        self.growth_rate = growth_rate
        self.p_drop = p_drop

        self.input_shape = input_shape
        self.depth = depth
        self.in_depth = in_depth
        self.n_classes = n_classes
        self.dtype = tf.keras.mixed_precision.experimental.Policy(float16_dtype) if float16 else tf.float32
        self.Block = Block
        self.Conv = Conv
        self.Activation = Activation
        self.expansion = expansion
        self.compression_rate = compression_rate
        self.reduce_kernel_size = reduce_kernel_size

    def get_layer(self, input_layer):
        x = self.Conv(self.growth_rate*2, (3, 3), padding='same', use_bias=False, name="stem", activation=None)(input_layer)

        for i in range(self.depth):
            x = self.dense_block(x, name=f"dense_block{i:02d}_")
            if (i+1) < self.depth:
                x = self.reduce_block(x, compression_rate=self.compression_rate, name=f"reduce_block{i:02d}")

        x = tf.keras.layers.BatchNormalization(name=f"out_bn", dtype=self.dtype)(x)
        x = self.Activation(name="out_act", dtype=self.dtype)(x)

        if self.n_classes > 0:
            x = SeparableConvBN(self.n_classes, (3, 3), strides=(2, 2), use_bias=False,
                       activation=self.Activation, name="out_sep_conv_bn", dtype=self.dtype)(x)
            x = tf.keras.layers.GlobalAveragePooling2D(name="out_avg_pool", dtype=self.dtype)(x)
            x = tf.keras.layers.Softmax(name="out_softmax")(x)

        return x

    def build_model(self, input_layer=None):
        if input_layer is None:
            input_layer = tf.keras.layers.Input(self.input_shape)

        layer = self.get_layer(input_layer)
        model = tf.keras.models.Model(input_layer, layer)

        return model

    def dense_block(self, x, name="dense_block"):
        layers = [x]

        for i in range(self.in_depth):
            x = self.Block(expansion=self.expansion, growth_rate=self.growth_rate, p_drop=self.p_drop,
                           Activation=self.Activation, Conv2D=self.Conv, dtype=self.dtype, name=f"{name}{i:02d}")(layers[-1])

            layers.append(x)

        layers = tf.keras.layers.Concatenate(name=f"{name}_out_concat", dtype=self.dtype)(layers)
        return layers

    def reduce_block(self, x, compression_rate=2, name="reduce_block"):
        n_filter = int(x.shape[-1] // compression_rate)
        x = DenseReduce(n_filter, downsample_kernel_size=self.reduce_kernel_size,
                        Activation=self.Activation, name=name, dtype=self.dtype)(x)
        return x








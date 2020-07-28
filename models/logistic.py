import tensorflow as tf


class MiniModel:
    def __init__(self, input_shape=(None, None, 3), Activation=tf.keras.layers.ReLU, n_classes=10, hidden_ratio=0.5,
                 float16=False, float16_dtype='mixed_float16'):
        self.input_shape = input_shape
        self.Activation = Activation
        self.n_classes = n_classes
        self.hidden_ratio = hidden_ratio
        self.dtype = tf.keras.mixed_precision.experimental.Policy(float16_dtype) if float16 else tf.float32

    def get_layer(self, input_layer):
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME')(input_layer)
        x = tf.keras.layers.Flatten()(x)

        n_neuron = x.shape[1]
        x = tf.keras.layers.Dense(int(n_neuron*self.hidden_ratio), activation=None, dtype=self.dtype)(x)
        x = self.Activation(dtype=self.dtype)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.expand_dims(x, axis=-1)

        x = tf.keras.layers.MaxPool1D(2, dtype=self.dtype)(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(self.n_classes, name="out_logit", dtype=self.dtype)(x)
        x = tf.keras.layers.Softmax(name="out_softmax")(x)

        return x

    def build_model(self, input_layer=None):
        if input_layer is None:
            input_layer = tf.keras.layers.Input(self.input_shape)

        layer = self.get_layer(input_layer)
        model = tf.keras.models.Model(input_layer, layer)

        return model

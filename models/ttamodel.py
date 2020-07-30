import tensorflow as tf


class TTAModel:
    def __init__(self, model, n_tta=3, use_softmax=False):
        input_shape = model.input.shape[1:].as_list()
        self.data_format = "channels_last" if model.input.shape[-1] == 3 else "channels_first"
        if self.data_format == "channels_last":
            input_shape[-1] *= n_tta
        else:
            input_shape[0] *= n_tta

        self.input_shape = input_shape
        self.n_tta = n_tta
        self.model = model
        self.use_softmax = use_softmax
        self.tta_model = None

    def compile(self, **kwargs):
        self.tta_model.compile(**kwargs)

    def save(self, filepath, **kwargs):
        self.model.save(filepath, **kwargs)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.split(input_layer, self.n_tta, axis=-1 if self.data_format == "channels_last" else 1)

        pred = []
        for i in range(self.n_tta):
            pred.append(tf.expand_dims(self.model(x[i]), axis=-1))

        pred = tf.concat(pred, axis=-1)
        pred = tf.reduce_sum(pred, axis=-1)

        if self.use_softmax:
            pred = tf.keras.layers.Softmax()(pred)

        self.tta_model = tf.keras.models.Model(input_layer, pred)

        return self.tta_model

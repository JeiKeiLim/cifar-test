import tensorflow as tf
from tfhelper.metrics import F1ScoreMetric


class EnsembleModel:
    def __init__(self, models, input_shape=(48, 64, 3)):
        self.data_format = "channels_last" if models[0].input.shape[-1] == 3 else "channels_first"
        self.input_shape = input_shape if self.data_format == "channels_last" else (input_shape[-1], ) + input_shape[:2]
        self.n_classes = models[0].output.shape[-1]

        self.macro_f1score = F1ScoreMetric(n_classes=self.n_classes, debug=False, name="macro_f1score", f1_method='macro')
        self.geometric_f1score = F1ScoreMetric(n_classes=self.n_classes, debug=False, name="geometric_f1score", f1_method='geometric')

        self.metrics = ['accuracy', self.macro_f1score, self.geometric_f1score]

        for i, model in enumerate(models):
            model._name = f"model_{i}"

        self.models = models
        self.input = tf.keras.layers.Input(shape=self.input_shape)
        self.outs = [model(self.input) for model in self.models]
        self.ensemble_out = tf.keras.layers.Softmax()(tf.keras.layers.Add()(self.outs))
        self.ensemble_model = tf.keras.models.Model(self.input, self.ensemble_out)
        self.dtype = tf.float32

    def build_model(self):
        return self.ensemble_model

    def compile(self, **kwargs):
        kwargs['metrics'] = self.metrics

        self.ensemble_model.compile(**kwargs)

        for model in self.models:
            model.compile(**kwargs)

    def evaluate(self, test_set, eval_ensemble=True):
        if eval_ensemble:
            self.ensemble_model.evaluate(test_set)
        else:
            for i, model in enumerate(self.models):
                print(f"\n{i:02d} :: Evaluating {model.name} ...")
                model.evaluate(test_set)


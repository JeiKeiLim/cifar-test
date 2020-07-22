import tensorflow as tf
from functools import partial


class DistillationModel:
    def __init__(self, teacher_model, student_model, temperature=2.0):
        self.input_shape = tuple(teacher_model.input.shape[1:])
        self.n_classes = teacher_model.output.shape[1]

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.teacher_loss = None
        self.teacher_accuracy = None
        self.distill_model = None

    def build_model(self):
        self.teacher_model.trainable = False
        model_out = tf.keras.layers.Concatenate(name="distill_out")([tf.expand_dims(self.student_model.output, axis=-1),
                                                   tf.expand_dims(self.teacher_model(self.student_model.input), axis=-1)])
        self.distill_model = tf.keras.models.Model(self.student_model.input, [self.student_model.output, model_out])

        return self.distill_model

    def evaluate_teacher(self, x_test=None, y_test=None, test_set=None, loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        assert (x_test and y_test) or test_set

        self.teacher_model.compile(loss=loss, metrics=metrics)

        if test_set:
            loss, accuracy = self.teacher_model.evaluate(test_set)
        else:
            loss, accuracy = self.teacher_model.evaluate(x_test, y_test)

        self.teacher_loss = loss
        self.teacher_accuracy = accuracy

    def compile(self, loss='sparse_categorical_crossentropy', **kwargs):
        metric_s_func = partial(self.metric_accuracy, student=True)
        metric_t_func = partial(self.metric_accuracy, student=False)
        metric_s_func.__name__ = 'accuracy_student'
        metric_t_func.__name__ = 'accuracy_teacher'

        kwargs['loss'] = [loss, self.loss_soft]
        kwargs['metrics'] = [metric_s_func, metric_t_func]

        self.distill_model.compile(**kwargs)

    def loss_soft(self, y_true, y_pred):
        student_pred, teacher_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        student_pred = tf.reshape(student_pred, tf.shape(student_pred)[:-1])
        teacher_pred = tf.reshape(teacher_pred, tf.shape(teacher_pred)[:-1])

        soft_label = tf.exp(teacher_pred / self.temperature) / tf.expand_dims(tf.reduce_sum(tf.exp(teacher_pred / self.temperature), axis=-1), axis=-1)

        soft_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(soft_label, student_pred))

        return soft_loss * tf.square(self.temperature)

    def metric_accuracy(self, y_true, y_pred, student=True):
        accuracy = -1
        if student and tf.shape(y_pred).shape == 2:
            accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))
        elif tf.shape(y_pred).shape == 3:
            student_pred, teacher_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            teacher_pred = tf.reshape(teacher_pred, tf.shape(teacher_pred)[:-1])
            student_pred = tf.reshape(student_pred, tf.shape(student_pred)[:-1])

            y_pred = student_pred if student else teacher_pred

            accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))

        return accuracy

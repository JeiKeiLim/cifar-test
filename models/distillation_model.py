import tensorflow as tf
from models import ResNet
import numpy as np


class DistillationModel:
    def __init__(self, teacher_model, student_model, temperature=2.0):
        self.input_shape = tuple(teacher_model.input.shape[1:])
        self.n_classes = teacher_model.output.shape[1]

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.teacher_loss = None
        self.teacher_accuracy = None

    def evaluate_teacher(self, x_test=None, y_test=None, test_set=None):
        assert (x_test and y_test) or test_set

        if test_set:
            loss, accuracy = self.teacher_model.evaluate(test_set)
        else:
            loss, accuracy = self.teacher_model.evaluate(x_test, y_test)

        self.teacher_loss = loss
        self.teacher_accuracy = accuracy

    def compile(self, **kwargs):
        kwargs['loss'] = self.loss_function
        if 'metrics' in kwargs.keys():
            if type(kwargs['metrics']) == list:
                kwargs['metrics'] += [self.metric_accuracy]
            else:
                kwargs['metrics'] = [kwargs['metrics'], self.metric_accuracy]
        else:
            kwargs['metrics'] = [self.metric_accuracy]

        self.student_model.compile(**kwargs)

    def loss_function(self, y_true, y_pred):
        hard_label, soft_label = tf.split(y_true, num_or_size_splits=2, axis=1)

        hard_label = tf.reshape(hard_label, [-1, hard_label.shape[-1]])
        soft_label = tf.reshape(soft_label, [-1, soft_label.shape[-1]])

        soft_label = tf.exp(soft_label / self.temperature) / tf.expand_dims(tf.reduce_sum(tf.exp(soft_label / self.temperature), axis=-1), axis=-1)

        hard_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(hard_label, y_pred))
        soft_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(soft_label, y_pred))

        return hard_loss + tf.square(self.temperature) * soft_loss

    def metric_accuracy(self, y_true, y_pred):
        hard_label, soft_label = tf.split(y_true, num_or_size_splits=2, axis=1)

        hard_label = tf.reshape(hard_label, [-1, hard_label.shape[-1]])

        return tf.keras.metrics.categorical_accuracy(hard_label, y_pred)


class DistillationGenerator:
    def __init__(self, base_generator, n_data, teacher_model, image_size=(32, 32), from_teacher=True):
        self.base_generator = base_generator
        self.n_data = n_data
        self.teacher_model = teacher_model
        self.image_size = image_size
        self.from_teacher = from_teacher
        self.n_class = teacher_model.output.shape[1]

    def __call__(self):
        for i in range(self.n_data):
            d, l = next(self.base_generator)

            if self.from_teacher:
                teacher_result = self.teacher_model.predict(np.expand_dims(d, axis=0))

                l = tf.keras.utils.to_categorical([l], num_classes=self.n_class)
                l = np.concatenate([l, teacher_result], axis=0)

            yield d, l

    def get_tf_dataset(self, batch_size, shuffle=False, reshuffle=True, shuffle_size=64):
        out_shape = [2, self.n_class] if self.from_teacher else []
        out_dtype = tf.float32 if self.from_teacher else tf.int32

        dataset = tf.data.Dataset.from_generator(self,
                                                 (tf.float32, out_dtype),
                                                 (tf.TensorShape([self.image_size[0], self.image_size[1], 3]),
                                                  tf.TensorShape(out_shape))).batch(batch_size)

        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=reshuffle) if shuffle else dataset
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    from cifar_generator import CifarGenerator

    teacher_model = tf.keras.models.load_model("../saved_models/resnet152v2_cifar10_0.82.h5")
    student_model = ResNet(input_shape=(32, 32, 3), n_classes=10, n_layer=18).build_model()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_gen = CifarGenerator(x_train, y_train.flatten(), augment=True, model_type="resnetv2", image_size=(32, 32))
    test_gen = CifarGenerator(x_test, y_test.flatten(), augment=False, model_type="resnetv2", image_size=(32, 32))

    train_distill_gen = DistillationGenerator(train_gen(), train_gen.data.shape[0], teacher_model, from_teacher=True)
    test_distill_gen = DistillationGenerator(test_gen(), test_gen.data.shape[0], teacher_model, from_teacher=True)

    train_set = train_distill_gen.get_tf_dataset(32, shuffle=True, reshuffle=True, shuffle_size=12)
    test_set = test_distill_gen.get_tf_dataset(32, shuffle=False)

    resnet_distill = DistillationModel(teacher_model, student_model, temperature=2.0)
    resnet_distill.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    resnet_distill.student_model.fit(train_set, epochs=1, validation_data=test_set)

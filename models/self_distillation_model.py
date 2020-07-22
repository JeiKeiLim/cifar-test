import tensorflow as tf
from models import BottleNeckBlock, ConvBN
from models import ResNet, ResNet18, ResNet10
from functools import partial


class SelfDistillationModel:
    def __init__(self, target_model, out_layer_names, final_n_filter, final_feat_layer_name, out_layer_strides=[8, 4, 2], temperature=2.0, feat_lambda=0.001):
        self.model = target_model
        self.temperature = temperature
        self.final_n_filter = final_n_filter
        self.out_layer_names = out_layer_names
        self.out_layer_strides = out_layer_strides
        self.final_feat_layer_name = final_feat_layer_name
        self.n_class = self.model.output.shape[1]
        self.feat_lambda = feat_lambda
        self.n_out = (len(out_layer_names)+1) * 2
        self.distill_model = None

    def build_model(self):
        distill_outs_feat = []
        for i, (layer_name, stride) in enumerate(zip(self.out_layer_names, self.out_layer_strides)):
            distill_out = BottleNeckBlock(self.final_n_filter,
                                          (stride*3+1, stride*3+1),
                                          downsample=True,
                                          out_filter_multiplier=1,
                                          downsample_strides=(stride, stride),
                                          name="distill_bottleneck{:02d}".format(i))(self.model.get_layer(layer_name).output)
            distill_outs_feat.append(distill_out)
        distill_outs_feat.append(self.model.get_layer(self.final_feat_layer_name).output)

        distill_outs_logit = []
        for i in range(len(distill_outs_feat)-1):
            distill_out = ConvBN(self.n_class, (1, 1), name="distill_outconv{:02d}".format(i))(distill_outs_feat[i])
            distill_out = tf.keras.layers.GlobalAveragePooling2D(name="distill_avgpool{:02d}".format(i))(distill_out)
            distill_out = tf.keras.layers.Softmax(name="distill_softmax{:02d}".format(i))(distill_out)

            distill_outs_logit.append(distill_out)

        distill_outs_logit.append(self.model.output)

        for i in range(len(distill_outs_feat)):
            distill_outs_feat[i] = tf.expand_dims(distill_outs_feat[i], axis=-1)
            distill_outs_logit[i] = tf.expand_dims(distill_outs_logit[i], axis=-1)

        distill_outs_feat = tf.keras.layers.Concatenate(name="out_feats")(distill_outs_feat)
        distill_outs_logit = tf.keras.layers.Concatenate(name="out_logits")(distill_outs_logit)

        self.distill_model = tf.keras.models.Model(self.model.input, [self.model.output, distill_outs_feat, distill_outs_logit])

        return self.distill_model

    def compile(self, **kwargs):
        metrics = [
            self.metric_out_accuracy,
        ]
        for i in range((self.n_out//2)-1):
            p_func = partial(self.metric_feat_loss, i=i)
            p_func.__name__ = f"feat_metric_{i}"
            metrics.append(p_func)

        for i in range(self.n_out//2):
            p_func = partial(self.metric_logit_accuracy, i=i)
            p_func.__name__ = f"logit_metric_{i}"
            metrics.append(p_func)

        losses = [self.loss_out, self.loss_feat, self.loss_logit]

        kwargs['metrics'] = metrics
        kwargs['loss'] = losses
        kwargs['loss_weights'] = [1.0, self.feat_lambda, 1.0]

        self.distill_model.compile(**kwargs)

    def loss_out(self, y_true, y_pred):
        out_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
        return out_loss

    def loss_feat(self, y_true, y_pred):
        y_pred = tf.split(y_pred, self.n_out//2, axis=-1)
        feat_true, feat_pred = y_pred[-1], y_pred[:-1]

        feat_loss = 0
        for i in range((self.n_out//2)-1):
            feat_loss += tf.reduce_mean(tf.keras.losses.mean_squared_error(feat_true, feat_pred[i]))

        return feat_loss

    def loss_logit(self, y_true, y_pred):
        y_pred = tf.split(y_pred, self.n_out//2, axis=-1)
        logit_true, logit_pred = y_pred[-1], y_pred[:-1]

        logit_true = tf.reshape(logit_true, tf.shape(logit_true)[:-1])
        for i in range((self.n_out // 2) - 1):
            logit_pred[i] = tf.reshape(logit_pred[i], tf.shape(logit_pred[i])[:-1])

        logit_loss = 0
        soft_logit = tf.exp(logit_true/self.temperature) / tf.expand_dims(tf.reduce_sum(tf.exp(logit_true / self.temperature), axis=-1), axis=-1)

        for i in range((self.n_out//2)-1):
            logit_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(soft_logit, logit_pred[i]))

        return logit_loss

    def metric_out_accuracy(self, y_true, y_pred):
        if tf.shape(y_pred).shape == 2:
            return tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))
        else:
            return -1

    def metric_feat_loss(self, y_true, y_pred, i=0):
        feat_loss = -1
        if tf.shape(y_pred).shape == 5:
            y_pred = tf.split(y_pred, self.n_out // 2, axis=-1)
            feat_true, feat_pred = y_pred[-1], y_pred[:-1]
            feat_true = tf.reshape(feat_true, tf.shape(feat_true)[:-1])
            feat_pred = tf.reshape(feat_pred[i], tf.shape(feat_pred[i])[:-1])
            feat_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(feat_true, feat_pred))

        return feat_loss

    def metric_logit_accuracy(self, y_true, y_pred, i=0):
        logit_loss = -1
        if tf.shape(y_pred).shape == 3:
            y_pred = tf.split(y_pred, self.n_out // 2, axis=-1)
            y_pred = tf.reshape(y_pred[i], tf.shape(y_pred[i])[:-1])

            logit_loss = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))

        return logit_loss


if __name__ == "__main__":
    distill_param_dict = {
        tf.keras.applications.ResNet50: (['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out'], 2048, 'conv5_block3_out'),
        ResNet18: (['resblock_0_1_activation_1', 'resblock_1_1_activation_1', 'resblock_2_1_activation_1'], 512, 'resblock_3_1_activation_1')
    }

    out_layer_names , final_n_filter, final_feat_layer_name = distill_param_dict[ResNet18]
    model = ResNet18(input_shape=(32, 32, 3), n_classes=10, include_top=True).build_model()

    self_distiller = SelfDistillationModel(model, out_layer_names , final_n_filter, final_feat_layer_name, temperature=2.0)
    self_distiller.build_model()
    self_distiller.compile(optimizer='adam')
    distill_model = self_distiller.distill_model

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    distill_model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))


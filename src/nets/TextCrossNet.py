import tensorflow as tf
from tensorflow import keras


class TextCrossNet:
    def __init__(
            self,
            doc_len,
            block_len,
            feature_len,
            emb_dim,
            n_feature,
            n_word,
            emb_pretrain=[],
            l2=0,
            mode='cross',
            top_k=5,
            cate_emb='diag',
            dropout=0,
    ):

        self.doc_len = doc_len
        self.block_len = block_len
        self.emb_dim = emb_dim
        self.feature_len = feature_len
        self.n_features = n_feature
        self.n_word = n_word
        self.l2 = l2
        self.top_k = top_k
        self.training = tf.placeholder(dtype=tf.bool, name='training')

        if len(emb_pretrain) > 0:
            def myinit(*args, **kwargs):
                return tf.convert_to_tensor(emb_pretrain, dtype=tf.float32)
            self.emb_init = myinit
        else:
            self.emb_init = 'RandomNormal'

        jd = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='jds')
        cv = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='cvs')
        features = tf.placeholder(dtype=tf.int32, shape=(None, feature_len), name='cate_features')
        if mode == 'cate':
            features = self.feature_emb(features)
        else:
            with tf.variable_scope('jd_cnn'):
                jd = self.text_cnn(jd)
            with tf.variable_scope('cv_cnn'):
                cv = self.text_cnn(cv)
            with tf.variable_scope('cross'):
                if mode == 'text_concat':
                    features = self.concat_reduce(jd, cv)
                if mode == 'text_cross':
                    features = self.matmul_flatten(jd, cv)
                if mode == 'cross':
                    features, self.related_feature = self.cross(jd, cv, features, cate_emb)
                if mode == 'concat':
                    features = self.concat(jd, cv, features)

        with tf.variable_scope('classifier'):
            self.predict = self.classifier(features, dropout)
        with tf.variable_scope('loss'):
            self.label = tf.placeholder(dtype=tf.int32, shape=None, name='labels')
            self.loss = self.loss_function()

    @staticmethod
    def concat_reduce(jd, cv):
        features = tf.concat([jd, cv], axis=2)
        features = tf.reduce_mean(features, axis=1)
        # features = tf.layers.flatten(features)
        return features

    @staticmethod
    def matmul_flatten(jd, cv):
        cross = tf.matmul(jd, cv, transpose_b=True)
        cross = tf.layers.flatten(cross)
        return cross

    def feature_emb(self, cate):
        cate = keras.layers.Embedding(
            input_dim=self.n_features,
            output_dim=self.emb_dim,
            embeddings_initializer='RandomNormal',
            # embeddings_regularizer=keras.regularizers.l1_l2(),
            name='cate_embedding'
        )(cate)
        cate = tf.layers.flatten(cate)
        return cate

    def concat(self, jd, cv, cate):
        features = self.concat_reduce(jd, cv)
        cate = self.feature_emb(cate)
        features = tf.concat([features, cate], axis=1)
        return features

    def text_cnn(self, x: tf.Tensor):
        x = keras.layers.Embedding(
                input_dim=self.n_word,
                output_dim=self.emb_dim,
                embeddings_initializer=self.emb_init,
                # embeddings_regularizer=keras.regularizers.l1_l2()
        )(x)
        x = tf.reshape(x, shape=(-1, self.doc_len, self.emb_dim, 1))
        x = tf.layers.conv2d(
            inputs=x,
            filters=1,
            kernel_size=[self.block_len, 1],
            padding='same',
            activation=tf.nn.relu,
            name='cnn',
        )
        x = tf.squeeze(x, axis=3)
        # x = tf.pad(x, paddings=((0, 0), (2, 2), (0, 0), (0, 0)))
        # x = tf.layers.conv2d(
        #     inputs=x,
        #     filters=self.emb_dim,
        #     kernel_size=[5, self.emb_dim],
        #     activation=tf.nn.relu,
        #     name='cnn',
        #     # kernel_regularizer=keras.regularizers.l1_l2()
        # )
        # x = tf.squeeze(x, axis=2)
        return x

    def cross(self, jd: tf.Tensor, cv: tf.Tensor, cate: tf.Tensor, cate_emb='diag'):
        if cate_emb == 'diag':
            emb_dim = self.emb_dim
        else:
            emb_dim = self.emb_dim ** 2
        cate = keras.layers.Embedding(
            input_dim=self.n_features,
            output_dim=emb_dim,
            embeddings_initializer='RandomNormal',
            # embeddings_regularizer=keras.regularizers.l1_l2(),
            name='cate_embedding'
        )(cate)
        if cate_emb == 'diag':
            cate = tf.matrix_diag(cate)
        else:
            cate = tf.reshape(cate, shape=[-1, self.feature_len, self.emb_dim, self.emb_dim])
        jd = tf.tile(
            tf.reshape(jd, shape=(-1, 1, self.doc_len, self.emb_dim)),
            multiples=(1, self.feature_len, 1, 1),
        )
        cv = tf.tile(
            tf.reshape(cv, shape=(-1, 1, self.doc_len, self.emb_dim)),
            multiples=(1, self.feature_len, 1, 1)
        )
        cross = tf.matmul(jd, cate)
        # cross = tf.nn.tanh(cross)
        cross = tf.matmul(cross, cv, transpose_b=True)
        cross = tf.reshape(cross, shape=[-1, self.feature_len, self.doc_len ** 2])
        cross = tf.nn.softmax(cross)
        cross = tf.layers.flatten(cross)
        related_features = tf.nn.top_k(cross, k=self.top_k)[1]
        return cross, related_features

    def classifier(self, features, dropout=0):
        if dropout:
            features = tf.layers.dropout(
                inputs=features,
                rate=dropout,
                training=self.training,
            )
        features = tf.layers.dense(
            features,
            units=self.emb_dim ** 2,
            activation=tf.nn.relu,
            name='hidden1'
        )
        if dropout:
            features = tf.layers.dropout(
                inputs=features,
                rate=dropout,
                training=self.training,
            )
        features = tf.layers.dense(
            features,
            units=self.emb_dim,
            activation=tf.nn.relu,
            name='hidden2'
        )
        if dropout:
            features = tf.layers.dropout(
                inputs=features,
                rate=dropout,
                training=self.training,
            )
        predict = tf.layers.dense(
            features,
            units=1,
            activation=tf.nn.sigmoid,
            name='output',
            # kernel_regularizer=keras.regularizers.l1_l2()
        )
        return predict

    def loss_function(self):
        predict = tf.squeeze(self.predict)
        loss = tf.losses.log_loss(self.label, predict)
        if self.l2:
            l2_loss = sum([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
            loss = loss + l2_loss * self.l2
        return loss

    def estimator_function(self):
        def function(features, labels, mode):
            jd = features["jd"]
            jd = self.text_cnn(jd)
            cv = features["cv"]
            cv = self.text_cnn(cv)
            cate_feature = features["cate"]
            cross_feature, _ = self.cross(jd, cv, cate_feature)
            logits = self.classifier(cross_feature)
            predictions = {
                "classes": tf.round(logits),
                "probabilities": logits,
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            loss = tf.losses.log_loss(labels=labels, predictions=logits)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return function


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    path = "board_test_network"
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    with tf.Session(graph=tf.Graph()) as sess:
        writer = tf.summary.FileWriter(path)
        text_cross_net = TextCrossNet(100, 5, 15, 128, 5000, 20000)
        # est_fn = text_cross_net.estimator_function()
        # text_cross_estimator = tf.estimator.Estimator(est_fn)
        writer.add_graph(sess.graph)
    writer.close()

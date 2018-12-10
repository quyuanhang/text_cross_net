import tensorflow as tf
from tensorflow import keras

class TextCrossNet:
    def __init__(self, doc_len, emb_dim, n_feature, n_word):
        self.doc_len = doc_len
        self.emb_dim = emb_dim
        self.n_features = n_feature
        self.n_word = n_word

        with tf.variable_scope('jd_cnn'):
            jd = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='jd')
            jd = self.text_cnn(jd)
        with tf.variable_scope('cv_cnn'):
            cv = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='cv')
            cv = self.text_cnn(cv)
        with tf.variable_scope('cross'):
            features = tf.placeholder(dtype=tf.int32, shape=(None, n_feature), name='cate_features')
            features, self.cross_related_feature = self.cross(jd, cv, features)
        with tf.variable_scope('classifier'):
            self.predict = self.linear(features)
        with tf.variable_scope('loss'):
            self.label = tf.placeholder(dtype=tf.int32, shape=None, name='label')
            self.loss = self.loss_function()

    def text_cnn(self, x:tf.Tensor):
        x = keras.layers.Embedding(
                input_dim=self.n_word,
                output_dim=self.emb_dim,
                embeddings_initializer='RandomNormal',
        )(x)
        x = tf.reshape(x, shape=(-1, self.doc_len, self.emb_dim, 1))
        x = tf.layers.conv2d(
            inputs=x,
            filters=1,
            kernel_size=[3, 1],
            padding='same',
            activation=tf.nn.relu,
            name='cnn',
        )
        x = tf.squeeze(x)
        return x

    def cross(self, jd: tf.Tensor, cv: tf.Tensor, cate: tf.Tensor):
        cate = keras.layers.Embedding(
            input_dim=self.n_features,
            output_dim=self.emb_dim,
            embeddings_initializer='RandomNormal',
            name='cate_embedding'
        )(cate)
        # cate = tf.map_fn(
        #     tf.matrix_diag,
        #     cate,
        #     name='diag'
        # )
        cate = tf.matrix_diag(cate)
        jd = tf.tile(
            tf.reshape(jd, shape=(-1, 1, self.doc_len, self.emb_dim)),
            multiples=(1, self.n_features, 1, 1),
        )
        cv = tf.tile(
            tf.reshape(cv, shape=(-1, 1, self.doc_len, self.emb_dim)),
            multiples=(1, self.n_features, 1, 1)
        )
        cross = tf.matmul(jd, cate)
        cross = tf.matmul(cross, cv, transpose_b=True)
        cross = tf.reduce_max(cross, axis=1)
        related_features = tf.argmax(cross, axis=1)
        cross = tf.layers.flatten(cross)
        return cross, related_features

    @staticmethod
    def linear(features):
        predict = tf.layers.dense(
            features,
            units=1,
            activation=tf.nn.sigmoid,
            name='output'
        )
        return predict

    def loss_function(self):
        predict = tf.squeeze(self.predict)
        loss = tf.losses.log_loss(self.label, predict)
        return loss


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    path = "test_network"
    if os.path.exists(path):
        os.rmdir(path)
    with tf.Session(graph=tf.Graph()) as sess:
        writer = tf.summary.FileWriter(path)
        text_cross_net = TextCrossNet(100, 128, 10, 20000)
        writer.add_graph(sess.graph)
    writer.close()

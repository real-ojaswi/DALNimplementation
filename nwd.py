from grl import WarmStartGradientReverseLayer
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class NuclearWassersteinDiscrepancy(Loss):
    def __init__(self, classifier, trade_off_lambda=1.0):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.classifier = classifier
        self.grl = WarmStartGradientReverseLayer()
        self.trade_off_lambda = trade_off_lambda

    def n_discrepancy(self, y_s, y_t):
        pre_s, pre_t = tf.nn.softmax(y_s, axis=1), tf.nn.softmax(y_t, axis=1)  # Work on this later
        u_t, s_t, v_t = tf.linalg.svd(pre_t)
        u_s, s_s, v_s = tf.linalg.svd(pre_s)
        nuclear_norm_t = tf.reduce_sum(s_t)
        nuclear_norm_s = tf.reduce_sum(s_s)
        loss = -nuclear_norm_t / tf.cast(tf.shape(y_t)[0], tf.float32) + nuclear_norm_s / tf.cast(tf.shape(y_s)[0],
                                                                                                  tf.float32)
        return loss

    def call(self, f):
        f_grl = self.grl(f)
        y = self.classifier(f_grl, training=True)
        y_s, y_t = tf.split(y, num_or_size_splits=2, axis=0)

        loss = self.n_discrepancy(y_s, y_t)
        return loss
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from nwd import NuclearWassersteinDiscrepancy
class Model():
    def __init__(self):

        self.feature_extractor = Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, input_shape=(32,32,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=48, kernel_size=5, strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten()
        ])

        self.label_predictor = Sequential([
            tf.keras.layers.Dense(100),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(100),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.predict_label=Sequential([self.feature_extractor, self.label_predictor])

        self.lp_optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.005, decay=0.001, momentum=0.9)
        self.fe_optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.0005, decay=0.001, momentum=0.9)






    @tf.function
    def train_step(self, x_source, x_target, y):
            x=tf.concat([x_source, x_target], axis=0)#x_source, x_target = tf.split(inputs, num_or_size_splits=2, axis=0)
            with tf.GradientTape(persistent=True) as tape1:
                x_intermediate = self.feature_extractor(x_source, training=True)
                y_pred=self.label_predictor(x_intermediate, training=True)
                lp_loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)

            fe_grad1 = tape1.gradient(lp_loss, self.feature_extractor.trainable_variables)
            lp_grad1 = tape1.gradient(lp_loss, self.label_predictor.trainable_variables)

            with tf.GradientTape(persistent = True) as tape2:
                x=self.feature_extractor(x, training=True)
                discrepancy= NuclearWassersteinDiscrepancy(self.label_predictor, trade_off_lambda=1.0)
                divergence_loss = -discrepancy.call(x)

            lp_grad2 = tape2.gradient(divergence_loss, self.label_predictor.trainable_variables)
            fe_grad2 = tape2.gradient(divergence_loss, self.feature_extractor.trainable_variables)


            self.lp_optimizer.apply_gradients(zip(lp_grad1, self.label_predictor.trainable_variables))
            self.fe_optimizer.apply_gradients(zip(fe_grad1, self.feature_extractor.trainable_variables))

            self.lp_optimizer.apply_gradients(zip(lp_grad2, self.label_predictor.trainable_variables))
            self.fe_optimizer.apply_gradients(zip(fe_grad2, self.feature_extractor.trainable_variables))
            del tape1
            del tape2

            return lp_loss+divergence_loss




    @tf.function
    def train_source_only(self, x_source, y):
            with tf.GradientTape(persistent=True) as tape1:
                x_intermediate = self.feature_extractor(x_source, training=True)
                y_pred=self.label_predictor(x_intermediate, training=True)
                lp_loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)

            fe_grad1 = tape1.gradient(lp_loss, self.feature_extractor.trainable_variables)
            lp_grad1 = tape1.gradient(lp_loss, self.label_predictor.trainable_variables)


            self.lp_optimizer.apply_gradients(zip(lp_grad1, self.label_predictor.trainable_variables))
            self.fe_optimizer.apply_gradients(zip(fe_grad1, self.feature_extractor.trainable_variables))


            del tape1

            return lp_loss


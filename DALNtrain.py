import numpy as np
import pandas as pd
from DALNModel import Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from sklearn.metrics import accuracy_score
import cv2


class train():
    def __init__(self, X_source, y_source, batch_size=64, epochs=20, model=Model(), X_target=None, source_only=False):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.source_only = source_only
        self.__call__()

    def __call__(self):
        list_avg_losses=[]
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = len(self.X_source) // self.batch_size  # Adjust batch_size as needed
            for batch in range(num_batches):
                start_idx = int(batch * self.batch_size / 2)
                end_idx = int((batch + 1) * self.batch_size / 2)
                batch_inputs_s = self.X_source[start_idx:end_idx]
                batch_labels = self.y_source[start_idx:end_idx]
                # batch_inputs=tf.concat([batch_inputs_s, batch_inputs_t], axis=0)
                if self.source_only:
                    batch_loss = self.model.train_source_only(batch_inputs_s, batch_labels)
                else:
                    batch_inputs_t = self.X_target[start_idx:end_idx]
                    batch_loss = self.model.train_step(x_source=batch_inputs_s, x_target=batch_inputs_t, y=batch_labels)
                epoch_loss += batch_loss.numpy()

            # Calculate and display average loss for the epoch
            average_loss = epoch_loss / num_batches
            print(f'Epoch {epoch + 1} Loss: {np.mean(average_loss)}')
            list_avg_losses.append(average_loss)

        return {'final loss':average_loss, 'all avg losses': list_avg_losses}



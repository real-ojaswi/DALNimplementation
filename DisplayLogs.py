import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from DALNModel import Model
from DALNtrain import train
class display_logs():
    def __init__(self, X_source, y_source, batch_size=64, epochs=20, model=Model(), X_target=None, y_target=None,
                 source_only=False):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.y_target = y_target
        self.source_only = source_only
        self.trainer = train(X_source=self.X_source, y_source=self.y_source, model=self.model, batch_size=32,
                             X_target=self.X_target, epochs=self.epochs, source_only=self.source_only)
        self.y_predicted_prob_source = self.model.predict_label(self.X_source)
        if self.X_target is not None:
            self.y_predicted_prob_target = self.model.predict_label(self.X_target)

    def accuracy(self):
        y_predicted_source = np.argmax(self.y_predicted_prob_source, axis=1)
        accuracy_score_source = accuracy_score(y_predicted_source, self.y_source)
        if self.y_target is not None and self.X_target is not None:
            y_predicted_target = np.argmax(self.y_predicted_prob_target, axis=1)
            accuracy_score_target = accuracy_score(y_predicted_target, self.y_target)
            accuracy_log = {'accuracy_score_source': accuracy_score_source,
                            'accuracy_score_target': accuracy_score_target}
        else:
            accuracy_log = {'accuracy_score_source': accuracy_score_source}
        return accuracy_log

    def plot_tsne(self, num_epoch=None):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
        if self.X_target is not None:
            inter_features = tf.concat([self.y_predicted_prob_source, self.y_predicted_prob_target], axis=0)
        else:
            inter_features = self.y_predicted_prob_source

        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        projected_features = tsne.fit_transform(inter_features)

        # plotting
        if self.X_target is not None:
            colors = ['blue'] * len(self.X_source) + ['red'] * len(self.X_target)
        else:
            colors = ['blue'] * len(self.X_source)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            projected_features[:, 0],
            projected_features[:, 1],
            c=colors,
            s=2,  # Adjust marker size as needed
            alpha=0.7,  # Adjust alpha (transparency) if needed
        )
        if num_epoch is None:
            num_epoch = self.epochs
        plt.title(f't-SNE Plot Source vs Target at epoch: {num_epoch}')
        plt.show()

    def plot_pca(self, num_epoch=None):

        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import numpy as np
        if self.X_target is not None:
            X = tf.concat([self.X_source, self.X_target], axis=0)
            inter_features = self.model.feature_extractor(X)
        else:
            inter_features = self.model.feature_extractor(self.X_source)

        pca = PCA(n_components=2, random_state=42)
        projected_features = pca.fit_transform(inter_features)

        # plotting
        if self.X_target is not None:
            colors = ['blue'] * len(self.X_source) + ['red'] * len(self.X_target)
        else:
            colors = ['blue'] * len(self.X_source)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            projected_features[:, 0],
            projected_features[:, 1],
            c=colors,
            s=2,  # Adjust marker size as needed
            alpha=0.7,  # Adjust alpha (transparency) if needed
        )
        if num_epoch is None:
            num_epoch = self.epochs
        plt.title(f'PCA Plot Source vs Target at epoch {num_epoch}')
        plt.show()

    def plot_tsne_features(self, num_epoch=None):

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
        if self.X_target is not None:
            X = tf.concat([self.X_source[0:5000], self.X_target[0:5000]], axis=0)
            inter_features = self.model.feature_extractor(X)
        else:
            inter_features = self.model.feature_extractor(self.X_source[0:5000])

        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        projected_features = tsne.fit_transform(inter_features)

        # plotting
        if self.X_target is not None:
            colors = ['blue'] * len(self.X_source) + ['red'] * len(self.X_target)
        else:
            colors = ['blue'] * len(self.X_source)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            projected_features[:, 0],
            projected_features[:, 1],
            c=colors,
            s=2,  # Adjust marker size as needed
            alpha=0.7,  # Adjust alpha (transparency) if needed
        )
        if num_epoch is None:
            num_epoch = self.epochs
        plt.title(f'TSNE Plot Source vs Target at epoch {num_epoch}')
        plt.show()









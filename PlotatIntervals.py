from DisplayLogs import display_logs
from DALNModel import Model
class plot_tsne_at_intervals():
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

    def plot(self, intervals):
        for i in range(round(self.epochs / intervals)):
            logs_DA = display_logs(X_source=self.X_source, y_source=self.y_source, model=self.model,
                                   batch_size=self.batch_size, X_target=self.X_target, y_target=self.y_target,
                                   epochs=intervals, source_only=self.source_only)
            logs_DA.plot_tsne(num_epoch=(i + 1) * intervals)

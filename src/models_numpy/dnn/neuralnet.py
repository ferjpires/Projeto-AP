import numpy as np

from src.models_numpy.dnn.optimizer import SGDOptimizer

class NeuralNetwork:
    def __init__(
        self,
        epochs=100,
        batch_size=128,
        optimizer=None,
        verbose=False,
        loss=None,
        metric=None,
        early_stopping=True,
        patience=15,
        min_delta=0.001,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer if optimizer is not None else SGDOptimizer()
        self.verbose = verbose
        if loss is None:
            raise ValueError("A loss function logic must be provided (e.g., loss=CategoricalCrossEntropy)")
        self.loss = loss() if isinstance(loss, type) else loss
        self.metric = metric
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self.layers = []
        self.history = {}
        self.best_val_loss = np.inf
        self.no_improvement_count = 0
        self.best_weights = None
        self.best_epoch = None

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, "initialize"):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, (
            "Batch size cannot be greater than the number of samples"
        )
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            if y is not None:
                yield (
                    X[indices[start:end]],
                    y[indices[start:end]],
                )
            else:
                yield X[indices[start:end]], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def save_model_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, "get_weights"):
                weights.append(layer.get_weights())
            else:
                weights.append(None)
        return weights

    def restore_model_weights(self, weights):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "set_weights") and weights[i] is not None:
                layer.set_weights(weights[i])

    def fit(self, dataset, val_dataset=None):
        X = dataset.X
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        self.best_val_loss = np.inf
        self.best_epoch = None
        self.no_improvement_count = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, self.epochs + 1):
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            loss = self.loss.loss(y_all, output_x_all)

            if self.metric is not None:
                metric = self.metric(y_all, output_x_all)
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric_s = "NA"
                metric = "NA"

            self.history[epoch] = {"loss": loss, "metric": metric}
            history["train_loss"].append(loss)
            if metric != "NA":
                history["train_acc"].append(metric)
            else:
                history["train_acc"].append(0)

            val_loss, val_metric = None, None
            if val_dataset:
                val_X, val_y = val_dataset.X, val_dataset.y
                if np.ndim(val_y) == 1:
                    val_y = np.expand_dims(val_y, axis=1)

                val_output = self.forward_propagation(val_X, training=False)
                val_loss = self.loss.loss(val_y, val_output)

                if self.metric is not None:
                    val_metric = self.metric(val_y, val_output)
                    val_metric_s = f"{self.metric.__name__}: {val_metric:.4f}"
                else:
                    val_metric_s = "NA"

                self.history[epoch]["val_loss"] = val_loss
                self.history[epoch]["val_metric"] = val_metric
                history["val_loss"].append(val_loss)
                if val_metric is not None and val_metric != "NA":
                    history["val_acc"].append(val_metric)
                else:
                    history["val_acc"].append(0)

                if self.early_stopping:
                    if val_loss < (self.best_val_loss - self.min_delta):
                        self.best_val_loss = val_loss
                        self.no_improvement_count = 0
                        self.best_weights = self.save_model_weights()
                        self.best_epoch = epoch
                    else:
                        self.no_improvement_count += 1
                        if self.no_improvement_count >= self.patience:
                            if self.verbose:
                                print(
                                    f"Early stopping at epoch {epoch}. Best validation loss: {self.best_val_loss:.4f}"
                                )
                            if self.best_weights:
                                self.restore_model_weights(self.best_weights)
                            break

                if self.verbose:
                    print(
                        f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s} - val_loss: {val_loss:.4f} - val_{val_metric_s}"
                    )
            else:
                if self.verbose:
                    print(
                        f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}"
                    )

        if self.early_stopping and self.best_weights:
            self.restore_model_weights(self.best_weights)

        return history

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        else:
            raise ValueError("No metric specified for the neural network.")

    def save(self, file_path):
        model_data = {
            "layers": self.layers,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metric": self.metric,
        }
        np.savez(file_path, **model_data)

    def load(self, file_path):
        model_data = np.load(file_path, allow_pickle=True)
        self.layers = model_data["layers"]
        self.optimizer = model_data["optimizer"]
        self.loss = model_data["loss"]
        self.metric = model_data["metric"]

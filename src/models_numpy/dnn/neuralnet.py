import numpy as np
import pickle
from src.models_numpy.dnn.metrics import accuracy


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_func, optimizer):
        self.loss = loss_func.loss
        self.loss_prime = loss_func.derivative
        self.optimizer = optimizer

    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
            if hasattr(layer, 'weights'):
                self.optimizer.update(layer)

    def fit(self, X_train, y_train, epochs, val_data=None):
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            # Forward e Backward
            pred = self.forward(X_train, training=True)
            loss = self.loss(y_train, pred)
            acc = accuracy(y_train, pred)

            error = self.loss_prime(y_train, pred)
            self.backward(error)

            history['train_loss'].append(loss)
            history['train_acc'].append(acc)

            # Validação
            if val_data:
                X_val, y_val = val_data
                val_pred = self.forward(X_val, training=False)
                val_loss = self.loss(y_val, val_pred)
                val_acc = accuracy(y_val, val_pred)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} Acc: {acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} Acc: {acc:.4f}")

        return history

    def predict(self, X):
        out = self.forward(X, training=False)
        return np.argmax(out, axis=1)

    def save(self, filepath):
        model_data = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                model_data.append({'w': layer.weights, 'b': layer.biases})
            else:
                model_data.append(None)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        for i, layer in enumerate(self.layers):
            if model_data[i] is not None:
                layer.weights = model_data[i]['w']
                layer.biases = model_data[i]['b']

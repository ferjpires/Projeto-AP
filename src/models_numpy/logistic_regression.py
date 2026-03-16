import numpy as np


class LogisticRegression:
    """Regressão Logística Multinomial (Softmax Regression)."""
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01, epochs: int = 200):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, val_data=None, verbose: bool = True):
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        m = X.shape[0]

        for epoch in range(self.epochs):
            z = X @ self.weights + self.bias
            y_pred = self.softmax(z)

            dz = (y_pred - y) / m
            dw = X.T @ dz
            db = np.sum(dz, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = -np.sum(y * np.log(y_pred + 1e-12)) / m
            acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            history['train_loss'].append(loss)
            history['train_acc'].append(acc)

            if val_data is not None:
                X_val, y_val = val_data
                z_val = X_val @ self.weights + self.bias
                val_pred = self.softmax(z_val)
                val_loss = -np.sum(y_val * np.log(val_pred + 1e-12)) / y_val.shape[0]
                val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss:.4f} Acc: {acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss:.4f} Acc: {acc:.4f}")

        return history

    def predict(self, X):
        z = X @ self.weights + self.bias
        return np.argmax(self.softmax(z), axis=1)

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.softmax(z)

    def save(self, path: str):
        np.savez(path, weights=self.weights, bias=self.bias)

    def load(self, path: str):
        data = np.load(path)
        self.weights = data["weights"]
        self.bias = data["bias"]

import numpy as np


class CategoricalCrossEntropy:
    def loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / m

    def derivative(self, y_true, y_pred):
        # Derivada simplificada quando combinada com Softmax
        m = y_true.shape[0]
        return (y_pred - y_true) / m

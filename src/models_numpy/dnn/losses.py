import numpy as np
from abc import abstractmethod


class LossFunction:
    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class CategoricalCrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / m

    def derivative(self, y_true, y_pred):
        m = y_true.shape[0]
        return (y_pred - y_true) / m


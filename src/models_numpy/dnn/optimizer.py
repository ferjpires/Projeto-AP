import numpy as np
from abc import abstractmethod

class Optimizer:
    @abstractmethod
    def update(self, w, grad_loss_w):
        raise NotImplementedError

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.retained_gradient = None
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, w, grad_loss_w):
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = (
            self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        )
        return w - self.learning_rate * self.retained_gradient

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

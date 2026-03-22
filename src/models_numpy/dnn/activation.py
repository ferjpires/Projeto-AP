import numpy as np
from abc import abstractmethod
from src.models_numpy.dnn.layers import Layer


class ActivationLayer(Layer):
    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0


class ReLUActivation(ActivationLayer):
    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input <= 0, 0, 1)


class LeakyReLUActivation(ActivationLayer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activation_function(self, input):
        return np.where(input > 0, input, self.alpha * input)

    def derivative(self, input):
        return np.where(input > 0, 1, self.alpha)


class SoftmaxActivation(Layer):
    def forward_propagation(self, input, training):
        self.input = input
        exp_z = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output

    def backward_propagation(self, output_error):
        return output_error

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0

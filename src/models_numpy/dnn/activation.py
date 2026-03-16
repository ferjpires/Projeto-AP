import numpy as np
from src.models_numpy.dnn.layers import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data, training=True):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.activation_prime(self.input) * output_error


class ReLUActivation(ActivationLayer):
    def __init__(self):
        def relu(z):
            return np.maximum(0, z)
        def relu_prime(z):
            return (z > 0).astype(float)
        super().__init__(relu, relu_prime)


class SoftmaxActivation(Layer):
    def forward_propagation(self, input_data, training=True):
        self.input = input_data
        exp_z = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output

    def backward_propagation(self, output_error):
        return output_error

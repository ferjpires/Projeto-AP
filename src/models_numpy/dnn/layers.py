import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data, training):
        raise NotImplementedError

    def backward_propagation(self, output_error):
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Inicialização He (melhor para ReLU)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

        # Para o otimizador com momentum
        self.weights_momentum = np.zeros_like(self.weights)
        self.biases_momentum = np.zeros_like(self.biases)

    def forward_propagation(self, input_data, training=True):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        self.weights_error = np.dot(self.input.T, output_error)
        self.biases_error = np.sum(output_error, axis=0, keepdims=True)
        return input_error

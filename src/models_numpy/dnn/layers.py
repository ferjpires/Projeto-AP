import numpy as np
import copy
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__


class DenseLayer(Layer):
    def __init__(self, n_units, input_shape=None, l2_reg=0.001, max_norm=3.0):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.l2_reg = l2_reg
        self.max_norm = max_norm

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer):
        fan_in = self.input_shape()[0]
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.normal(0, std, (self.input_shape()[0], self.n_units))
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error) + self.l2_reg * self.weights
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        weights_error = np.clip(weights_error, -self.max_norm, self.max_norm)
        bias_error = np.clip(bias_error, -self.max_norm, self.max_norm)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self):
        return (self.n_units,)

    def get_weights(self):
        return {"weights": self.weights, "biases": self.biases}

    def set_weights(self, weights):
        self.weights = weights["weights"]
        self.biases = weights["biases"]


class DropoutLayer(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward_propagation(self, inputs, training):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (
                1 - self.rate
            )
            return inputs * self.mask
        return inputs

    def backward_propagation(self, output_error):
        return output_error * self.mask

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


class BatchNormalizationLayer(Layer):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

    def initialize(self, optimizer):
        input_shape = self.input_shape()
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)
        self.running_mean = np.zeros(input_shape)
        self.running_var = np.ones(input_shape)
        self.gamma_opt = copy.deepcopy(optimizer)
        self.beta_opt = copy.deepcopy(optimizer)

    def forward_propagation(self, input, training):
        self.input = input

        if training:
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)

            self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean
            self.running_var = 0.9 * self.running_var + 0.1 * batch_var

            self.x_norm = (input - batch_mean) / np.sqrt(batch_var + self.epsilon)
            self.output = self.gamma * self.x_norm + self.beta
        else:
            x_norm = (input - self.running_mean) / np.sqrt(
                self.running_var + self.epsilon
            )
            self.output = self.gamma * x_norm + self.beta

        return self.output

    def backward_propagation(self, output_error):
        gamma_grad = np.sum(output_error * self.x_norm, axis=0)
        beta_grad = np.sum(output_error, axis=0)

        self.gamma = self.gamma_opt.update(self.gamma, gamma_grad)
        self.beta = self.beta_opt.update(self.beta, beta_grad)

        batch_size = self.input.shape[0]
        x_mu = self.input - np.mean(self.input, axis=0)
        std_inv = 1.0 / np.sqrt(np.var(self.input, axis=0) + self.epsilon)

        dx_norm = output_error * self.gamma
        dvar = np.sum(dx_norm * x_mu, axis=0) * -0.5 * std_inv**3
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * x_mu, axis=0)

        input_error = (
            (dx_norm * std_inv) + (dvar * 2 * x_mu / batch_size) + (dmean / batch_size)
        )
        return input_error

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def get_weights(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def set_weights(self, weights):
        self.gamma = weights["gamma"]
        self.beta = weights["beta"]
        self.running_mean = weights["running_mean"]
        self.running_var = weights["running_var"]

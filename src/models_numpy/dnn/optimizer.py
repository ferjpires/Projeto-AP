class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, layer):
        if hasattr(layer, 'weights'):
            layer.weights_momentum = self.momentum * layer.weights_momentum + self.learning_rate * layer.weights_error
            layer.biases_momentum = self.momentum * layer.biases_momentum + self.learning_rate * layer.biases_error

            layer.weights -= layer.weights_momentum
            layer.biases -= layer.biases_momentum

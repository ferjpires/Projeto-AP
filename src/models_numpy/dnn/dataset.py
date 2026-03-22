import numpy as np


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def shuffle(self):
        indices = np.random.permutation(len(self))
        self.X = self.X[indices]
        self.y = self.y[indices]
        return self

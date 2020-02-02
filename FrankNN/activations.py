import numpy as np


class Activation:

    def __init__(self):
        pass

    def forawrd(self, x):
        pass

    def backward(self, dout):
        pass


class Linear(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout


class Sigmoid(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1 + np.exp(-x))

    def backward(self, dout):
        dx = (1 - dout) * dout
        return dx


class Relu(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        out = np.maximum(0, x)
        self.cache_x = x
        return out

    def backward(self, dout):
        dx = (self.cache_x > 0) * dout
        return dx


class Softmax(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        out = np.exp(x - np.max(x, axis=1, keepdims=True))
        out /= np.sum(out, axis=1, keepdims=True)
        return out

    def backward(self, dout):
        dx = (1 - dout) * dout
        return dx

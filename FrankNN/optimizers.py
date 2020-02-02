import numpy as np


class Optimizer:

    def __init__(self, learning_rate=1e-3):
        pass

    def update(self, w, dw, cache=None):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def update(self, w, dw, cache):
        w -= self.learning_rate * dw
        return w


class Adagrad(Optimizer):

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def update(self, w, dw, cache):
        if 'gti' not in cache:
            cache['gti'] = np.zeros(w.shape)

        gti = cache['gti']
        gti += dw**2
        dw_adjusted = dw / (1e-9 + np.sqrt(gti))
        w -= self.learning_rate * dw_adjusted
        cache['gti'] = gti
        return w

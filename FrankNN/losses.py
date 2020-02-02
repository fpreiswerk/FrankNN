import numpy as np


class Loss:

    def __init__(self):
        pass

    def get_loss(self, x, y):
        pass


class Softmax_cross_entropy_loss(Loss):

    def __init__(self):
        pass

    def get_loss(self, x, y):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

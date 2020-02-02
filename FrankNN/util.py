import numpy as np


def one_hot_encode(y):
    y_ = np.zeros((len(y), max(y) - min(y) + 1), dtype='int')
    for i in range(len(y)):
        y_[i, y[i]] = 1
    return y_


def reverse_one_hot_encode(y):
    return np.argmax(y, axis=1)

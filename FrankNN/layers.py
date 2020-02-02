import numpy as np
from .activations import *


class Layer:

    def __init__(self, shape):
        self.shape = shape
        self.w = self.b = self.x = None
        self.dw = self.db = self.dx = None
        self.cache_x = {}  # cache for backprop
        self.cache_w = {}  # cache for optimizer
        self.cache_b = {}  # cache for optimizer

    def forward(self, x):
        pass

    def backward(self, dout):
        pass

    def __str__(self):
        return 'Layer of shape ' + str((None,) + self.shape)

    def update_weights(self, update_rule):
        if self.w is not None:
            update_rule.update(self.w, self.dw, self.cache_w)
        if self.b is not None:
            update_rule.update(self.b, self.db, self.cache_b)


class Input(Layer):

    def __init__(self, shape):
        Layer.__init__(self, shape)

    def forward(self, x):
        return x


class Flatten(Layer):

    def __init__(self, input):
        Layer.__init__(self, (np.prod(input.shape),))

    def forward(self, x):
        self.cache_shape = x.shape
        return np.reshape(x, (x.shape[0], -1))

    def backward(self, dout):
        return np.reshape(dout, self.cache_shape)


class Dense(Layer):

    def __init__(self, input, channels, activation='linear'):
        Layer.__init__(self, (channels,))
        self.input = input
        self.w = np.clip(np.random.normal(
            0, 0.1, size=input.shape + (channels,)), -0.2, 0.2)
        self.b = np.ones(channels) * 0.1

        if activation == 'linear':
            self.activation = Linear()
        elif activation == 'relu':
            self.activation = Relu()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            raise ValueError('Unknown activation function')

    def forward(self, x):
        out = x.dot(self.w) + self.b
        self.cache_x = x
        return self.activation.forward(out)

    def backward(self, dout):
        dout = self.activation.backward(dout)
        self.dw = self.cache_x.T.dot(dout)
        self.db = np.sum(dout, axis=0)
        dxp = dout.dot(self.w.T)
        dx = dxp.reshape(self.cache_x.shape)
        return dx


class Conv2D(Layer):

    def __init__(self, input, kernel_size, channels, activation='linear',
                 stride=1):
        Layer.__init__(self, (input.shape[:-1] + (channels,)))
        W_shape = [kernel_size, kernel_size, input.shape[2], channels]
        self.w = np.clip(np.random.normal(0, 0.1, tuple(W_shape)), -0.2, 0.2)
        self.b = np.ones(channels) * 0.1
        self.stride = 1
        self.padding = int(np.ceil((kernel_size - 1) / 2))

        if activation == 'linear':
            self.activation = Linear()
        elif activation == 'relu':
            self.activation = Relu()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            raise ValueError('Unknown activation function')

    def forward(self, x):
        out = None

        N, height, width, C = x.shape
        weights = self.w
        hk, wk, C, F = weights.shape
        bias = self.b
        pad = self.padding

        out = np.zeros((N, height, width, F))
        x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                       'constant', constant_values=0)

        for n in range(N):  # for each image
            for f in range(F):  # for each output channel
                for j in range(height):
                    for i in range(width):
                        out[n, j, i, f] = \
                            np.sum(x_pad[n, j:j + hk, i:i + wk, :] *
                                   weights[:, :, :, f]) + bias[f]

        out = self.activation.forward(out)
        self.cache_x = x

        return out

    def backward(self, dout):
        dw, db = None, None

        x = self.cache_x
        w = self.w
        pad = self.padding
        stride = self.stride
        N, height, width, C = x.shape
        hk, wk, C, F = self.w.shape

        x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                       'constant', constant_values=0)

        # initialize gradients
        dx_pad = np.zeros_like(x_pad)
        dw = np.zeros((w.shape))
        db = np.sum(dout, axis=(0, 1, 2))

        dout = self.activation.backward(dout)

        for n in range(N):
            for f in range(F):
                for j in range(height):
                    for i in range(width):
                        dw[:, :, :, f] += x_pad[n, j * stride:j * stride + hk,
                                                i * stride:i * stride + wk, :] * dout[n, j, i, f]
                        dx_pad[n, j * stride:j * stride + hk, i * stride:i *
                               stride + wk, :] += w[:, :, :, f] * dout[n, j, i, f]
        dx = dx_pad[:, pad:pad + height, pad:pad + width, :]
        self.dw = dw
        self.db = db

        return dx


class MaxPooling(Layer):

    def __init__(self, input, pool_size=2):
        Layer.__init__(self, (input.shape[0] // 2,
                              input.shape[1] // 2, input.shape[2]))
        self.pool_size = pool_size

    def forward(self, x):
        self.cache_x = x
        ps = self.pool_size
        out = np.zeros((x.shape[0],) + self.shape)
        width = out.shape[2]
        height = out.shape[1]
        for j in range(height):
            for i in range(width):
                out[:, i, j, :] = np.max(x[:, i * ps:(i * ps + ps),
                                           j * ps:(j * ps + ps), :], axis=(1, 2))
        return out

    def backward(self, dout):
        x = self.cache_x
        ps = self.pool_size
        out = np.zeros((x.shape[0],) + self.shape)

        N = x.shape[0]
        C = x.shape[-1]
        width = dout.shape[1]
        height = dout.shape[2]
        dx = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for j in range(height):
                    for i in range(width):
                        ind = np.argmax(x[n, i * ps:(i * ps + ps),
                                          j * ps:(j * ps + ps), c])
                        ind1, ind2 = np.unravel_index(ind, (ps, ps))
                        dx[n, i:(i + 1 * ps), j:(j + 1 * ps),
                           :][ind1, ind2] = dout[n, j, i, c]
        return dx

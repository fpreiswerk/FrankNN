import numpy as np


class Model:

    def __init__(self):
        self.layers = []

    def __str__(self):
        ans = '{} Layers in model:'.format(len(self.layers))
        for n, layer in enumerate(self.layers):
            ans += '\nLayer {}: {}'.format(n, layer.__str__())
        return ans

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_output_layer(self):
        return self.layers[-1]

    def set_loss(self, loss):
        self.loss = loss

    def set_update_rule(self, update_rule):
        self.update_rule = update_rule

    def compute_loss(self, x, y):

        # do the forward pass
        out = self._forward(x)
        loss, dL = self.loss.get_loss(out, y)
        # compute all gradients through backprop
        self._backward(dL)
        return loss

    def predict(self, x):
        return self._forward(x)

    def _forward(self, x):

        x_ = x.copy()
        for i, layer in enumerate(self.layers):
            x_ = layer.forward(x_)
        return x_

    def _backward(self, x):

        x_ = x.copy()
        for i, layer in enumerate(self.layers[::-1]):
            x_ = layer.backward(x_)
        return x_

    def fit(self, x, y, batch_size, epochs, x_validate=None, y_validate=None,
            shuffle=True):

        print('Starting training with batch size {} for {} epochs'.format(
              batch_size, epochs))

        indices = np.arange(x.shape[0])
        if shuffle:
            np.random.shuffle(indices)
        n_batches_per_epoch = x.shape[0] // batch_size + 1

        for e in range(epochs):
            for b in range(n_batches_per_epoch):
                # sample next mini batch
                b_start = b*batch_size
                b_end = min((b+1)*batch_size, x.shape[0]-1)
                if not (b_start < b_end):
                    break
                x_batch = x[b*batch_size:(b+1)*batch_size, :]
                y_batch = y[b*batch_size:(b+1)*batch_size]

                # do a full forward-backward pass to obtain loss of this batch
                loss = self.compute_loss(x_batch, y_batch)
                status_msg = 'Epoch {}, mini batch {}, ' \
                             'training loss = {:.2f}'.format(e+1, b+1, loss)

                # update weights of all layers
                for layer in self.layers:
                    layer.update_weights(self.update_rule)

                # if a validation set is provided, feed it to the network
                if x_validate is not None:
                    val_loss = self.compute_loss(x_validate, y_validate)
                    status_msg += ', validation loss = {:.2f}'.format(val_loss)

                print(status_msg)

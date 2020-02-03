# FrankNN

This package imaplements all the basics for training neural networks using the backpropagation algorithm.
I originally wrote it as personal exercise, and now I'm using it for teaching others about neural nets.
The code is inspired by Andrej Karpathy's awesome cs231n course at Stanford [http://cs231n.github.io](http://cs231n.github.io) along with [some](https://github.com/MahanFathi/CS231) [great](https://github.com/Twice22/CS231n-solutions) [sources](https://github.com/benbo/adagrad/blob/master/adagrad.py). However, I decided to write my version of the code in a more object oriented way. This allows to define a network in just a few simple and almost self-explanatory lines of code, reminiscent of Keras and other high-level libraries:

```python
model = NN.Model()
model.add_layer(NN.Input(X.shape[1:]))
model.add_layer(NN.Dense(model.get_output_layer(), 16, activation='sigmoid'))
model.add_layer(NN.Dense(model.get_output_layer(), 32, activation='sigmoid'))
model.add_layer(NN.Dense(model.get_output_layer(), 64, activation='sigmoid'))
model.add_layer(NN.Dense(model.get_output_layer(), 3))
```

The above creates a vanilla 3 layer fully-connected network with 16, 32, and 64 hidden units, respectively, follwoed by 3 output classes.

The following components are currently implemented:

* Layers: dense, 2d convolution, max pooling
* Activations: linear, sigmoid, relu, softmax
* Loss functions: cross-entropy
* Optimizers: stochastic gradient descent (SGD), adagrad

*Disclaimer:* The focus of all code here is heavily on the educational side. In particular, the 2d convolutional layer is based on naive nested looping, and does not employ the well known im2col and col2im tricks. This is intentional, to make the code as easy to understand as possible.

## Examples

### Iris classification using a fully-connected neural network (FCN)

The program iris_dense_classifier.py implements a simple fully-connected network to classify the flowers in
the Iris dataset.

Example run:
```console
$ python iris_dense_classifier.py −−n_train 100 −−n_val 25 −−n_test 25 −−batch_size 150 −−epochs 30000
```

This should give the following output:
```console
Starting training with batch size 150 for 30000 epochs
Epoch 1, mini batch 1, training loss = 1.12, validation loss = 1.09
Epoch 2, mini batch 1, training loss = 1.11, validation loss = 1.09
...
Epoch 29999, mini batch 1, training loss = 0.23, validation loss = 0.27
Epoch 30000, mini batch 1, training loss = 0.23, validation loss = 0.27
Training took 0 minutes
Final test set accuracy = 96.00%
```

### MNIST classification using a convolutionan neural network (CNN)

The program mnist_cnn_classifier.py implements a simple convolutional neural network to classify the
MNIST digits. Since the code is neither optimized for speed nor running on the CPU, it is quite slow. 
Training for one epoch on 2000 samples takes about an hour on a 2015 MacBook Pro i7. 

Example run:
```console
$ python mnist_cnn_classifier.py −−n_train 2000 −−n_val 200 −−n_test 1000 −−batch_size 100 −−epochs 1
```

```console
Starting training with batch size 100 for 200 epochs
Epoch 1, mini batch 1, training loss = 7.15, validation loss = 11.55
Epoch 1, mini batch 2, training loss = 12.36, validation loss = 10.36
...
Epoch 1, mini batch 19, training loss = 2.37, validation loss = 1.51
Epoch 1, mini batch 20, training loss = 1.80, validation loss = 1.74
Training took 59 minutes
Final test set accuracy = 45.40%
```

Both training and validation loss show favorable convergence over this first epoch, but more epochs would be needed to
achieve higher accuracy. Accuracy is computed using a random sample of 1000 images from the test set. The initial learning rate used here was 0.001. I did not spend much time further optimizing the learning rate.




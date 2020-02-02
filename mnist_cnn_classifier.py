import argparse
import pickle
import time
import numpy as np
import gzip
import urllib.request
import FrankNN as NN

np.random.seed(42)

n_train = 5
n_validate = 5


def main(n_train, n_validate, n_test, batch_size, epochs):

    # download MNIST dataset
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training, validation, testing = pickle.load(f, encoding='latin1')

    x_train = np.reshape(training[0], [-1, 28, 28, 1])
    y_train = training[1]
    x_validate = np.reshape(validation[0], [-1, 28, 28, 1])
    y_validate = validation[1]
    x_test = np.reshape(testing[0], [-1, 28, 28, 1])
    y_test = testing[1]

    # select training, validation and test samples (randomly)
    train_inds = np.random.choice(x_train.shape[0],
                                  min(n_train, x_train.shape[0]))
    x_train = x_train[train_inds]
    y_train = y_train[train_inds]

    val_inds = np.random.choice(x_validate.shape[0],
                                min(n_validate, x_validate.shape[0]))
    x_validate = x_validate[val_inds]
    y_validate = y_validate[val_inds]

    test_inds = np.random.choice(x_test.shape[0], min(n_test, x_test.shape[0]))
    x_test = x_test[test_inds]
    y_test = y_test[test_inds]

    # assemble the model
    model = NN.Model()
    model.add_layer(NN.Input(x_train.shape[1:]))
    model.add_layer(NN.Conv2D(model.get_output_layer(), 10, 32,
                              activation='relu'))
    model.add_layer(NN.MaxPooling(model.get_output_layer()))
    model.add_layer(NN.Conv2D(model.get_output_layer(), 5, 16,
                              activation='relu'))
    model.add_layer(NN.MaxPooling(model.get_output_layer()))
    model.add_layer(NN.Flatten(model.get_output_layer()))
    model.add_layer(NN.Dense(model.get_output_layer(), 1024))
    model.add_layer(NN.Dense(model.get_output_layer(), 10))

    # specify loss and update rule
    model.set_loss(NN.Softmax_cross_entropy_loss())
    model.set_update_rule(NN.Adagrad())

    # start training
    start = time.time()
    model.fit(x_train, y_train,
              x_validate=x_validate,
              y_validate=y_validate,
              batch_size=batch_size, epochs=epochs)
    end = time.time()
    print('Training took {:.0f} minutes'.format((end - start)/60))

    # calculate final model accuracy on test set
    y_test_pred = model.predict(x_test)
    y_test_pred = NN.reverse_one_hot_encode(y_test_pred)
    accuracy = np.sum(y_test_pred == y_test) / y_test.shape[0]
    print('Final test set accuracy = {:.2f}%'.format(accuracy*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of mini batches')
    parser.add_argument('--n_train', type=int, default=50000,
                        help='Number of (random) training samples to use')
    parser.add_argument('--n_val', type=int, default=10000,
                        help='Number of (random) validation samples to use')
    parser.add_argument('--n_test', type=int, default=10000,
                        help='Number of (random) test samples to use')

    args = parser.parse_args()
    main(args.n_train, args.n_val, args.n_test, args.batch_size, args.epochs)

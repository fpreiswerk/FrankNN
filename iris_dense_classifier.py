import time
import numpy as np
import argparse
from sklearn import datasets
import FrankNN as NN


# Set a random seet for reproducibility.
np.random.seed(42)


def main(n_train, n_validate, n_test, batch_size, epochs):

    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    import pdb; pdb.set_trace()

    # Shuffle the data
    inds = np.arange(len(X))
    np.random.shuffle(inds)
    X = X[inds, :]
    y = y[inds]

    # Split the data into train, validation and test set
    x_train = X[0:n_train]
    y_train = y[0:n_train]
    x_validate = X[n_train:n_train + n_validate]
    y_validate = y[n_train:n_train + n_validate]
    x_test = X[-n_test:]
    y_test = y[-n_test:]

    # Assemble the model. A simple dataset like iris doesn't need a very deep model.
    # What worked well is an increasing number of channels with increasing
    # depth.
    model = NN.Model()
    model.add_layer(NN.Input(X.shape[1:]))
    model.add_layer(
        NN.Dense(
            model.get_output_layer(),
            16,
            activation='sigmoid'))
    model.add_layer(
        NN.Dense(
            model.get_output_layer(),
            32,
            activation='sigmoid'))
    model.add_layer(
        NN.Dense(
            model.get_output_layer(),
            64,
            activation='sigmoid'))
    model.add_layer(NN.Dense(model.get_output_layer(), 3))

    # Specify loss and update rule. I chose to use softmax cross entropy loss and
    # the adagrad update rule.
    model.set_loss(NN.Softmax_cross_entropy_loss())
    model.set_update_rule(NN.Adagrad(learning_rate=1e-3))

    # Start training.
    start = time.time()
    model.fit(x_train, y_train,
              x_validate=x_validate, y_validate=y_validate,
              batch_size=batch_size, epochs=epochs)
    end = time.time()
    print('Training took {:.0f} minutes'.format((end - start) / 60))

    # Calculate final model accuracy on test set.
    y_test_pred = model.predict(x_test)
    y_test_pred = NN.reverse_one_hot_encode(y_test_pred)
    accuracy = np.sum(y_test_pred == y_test) / y_test.shape[0]
    print('Final test set accuracy = {:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=150,
                        help='Size of mini batches')
    parser.add_argument('--n_train', type=int, default=100,
                        help='Number of (random) training samples to use')
    parser.add_argument('--n_val', type=int, default=25,
                        help='Number of (random) validation samples to use')
    parser.add_argument('--n_test', type=int, default=25,
                        help='Number of (random) test samples to use')

    args = parser.parse_args()
    main(args.n_train, args.n_val, args.n_test, args.batch_size, args.epochs)

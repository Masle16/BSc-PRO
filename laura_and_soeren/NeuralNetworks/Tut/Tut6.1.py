## A three-layer feedforward network consists of a perceptron hidden neuron
## -1 <= x1, x2 <= 1
import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


## Define training data
SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)

lr = 0.05
iters = 2500
no_experiments = 5
no_input = 2
no_output = 1
no_data = 11*11

hidden_neurons = [2, 4, 6, 8, 10]


    # Build feedforward network
def ffn(x, hidden_neurons):

        # Build input and output for hidden layer
    with tf.name_scope('hidden'):
        prut = tf.Variable(
            tf.truncated_normal( [no_input, hidden_neurons],
                                 stddev=1.0 / np.sqrt(float(no_input))),
            name='prut')

        biases = tf.Variable( tf.zeros( [hidden_neurons] ),
                               name='biases')

        h = tf.nn.sigmoid( tf.matmul( x, prut ) + biases )

        # Build input and output for output layer
    with tf.name_scope('linear'):   
        weights = tf.Variable(
            tf.truncated_normal( [hidden_neurons, no_output],
                                 stddev=1.0 / np.sqrt(float(hidden_neurons))),
            name='weights')
        biases = tf.Variable( tf.zeros( [1] ),
                               name='biases' )
        u = tf.matmul(h, weights) + biases

        # The output of the network
    return u

    # train the network and find errors
def train_exp(X, Y):

        # Split the data, 70:30
    x_train, y_train, x_test, y_test = X[:85], Y[:85], X[85:], Y[85:]

        # Container for error
    err = []

    for no_hidden in hidden_neurons:

            # Build the model
        x = tf.placeholder(tf.float32, [None, no_input])
        d = tf.placeholder(tf.float32, [None, no_output])

        y = ffn(x, no_hidden)

        loss = tf.reduce_mean( tf.reduce_sum( tf.square(d-y), axis=1))
        train = tf.train.GradientDescentOptimizer( lr ).minimize( loss )

            # Do the loops
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range( iters ):
                train.run( feed_dict={x:x_train, d:y_train} )
            err.append( loss.eval( feed_dict={x:x_test, d:y_test} ) )
            print(err)

    return err

def main():

        # Generate training data
    X_data = np.zeros((no_data, no_input))
    Y_data = np.zeros((no_data, no_output))

    i=0
    for x1 in np.arange(-1., 1.02, 0.2):
        print(x1)
        for x2 in np.arange(-1., 1.02, 0.2):
            X_data[i] = [x1,x2]
            Y_data[i] = np.sin(np.pi*x1)*np.cos(2*np.pi*x2)
            i += 1

    idx = np.arange(no_data)

        # Perform experiments
    err = []
    for exp in range(no_experiments):
        print('exp %d'%exp)

        np.random.shuffle(idx)
        err.append( train_exp(X_data[idx], Y_data[idx]))

        # Print the mean errors of different models
    mean_err = np.mean(np.array(err), axis=0)
    print(mean_err)
    

    plt.figure(1)
    plt.plot(hidden_neurons, mean_err, marker = 'x', linestyle = 'None')
    plt.xticks(hidden_neurons)
    plt.xlabel('number of hidden neurons')
    plt.ylabel('mean error')

    print(' *hidden units* %d '%hidden_neurons[np.argmin( mean_err )])

    plt.show()


if __name__ == '__main__':
    main()

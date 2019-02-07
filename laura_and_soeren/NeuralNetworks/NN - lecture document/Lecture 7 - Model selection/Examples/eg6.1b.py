#
# Chapter 6, example 1b
#

from sklearn import datasets
import numpy as np
import tensorflow as tf
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


no_iters = 1000
no_labels = 3
no_features = 4
no_exps = 25

hidden_units = [2, 4, 6, 8, 10, 12]

seed = 10

tf.set_random_seed(seed)
np.random.seed(seed)

def ffn(x, hidden_units):

  # Hidden 
  with tf.name_scope('hidden'):
    weights = tf.Variable(
      tf.truncated_normal([no_features, hidden_units],
                            stddev=1.0 / np.sqrt(float(no_features))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),
                         name='biases')
    hidden = tf.nn.relu(tf.matmul(x, weights) + biases)
    
  # output
  with tf.name_scope('linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units, no_labels],
                            stddev=1.0 / np.sqrt(float(hidden_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([no_labels]),
                         name='biases')
    logits = tf.matmul(hidden, weights) + biases
    
  return logits


def train_exp(X, Y):

    x_train, y_train, x_test, y_test = X[:100], Y[:100], X[100:], Y[100:]

    err = []
    for no_hidden in hidden_units:

        x = tf.placeholder(tf.float32, [None, no_features])
        y_ = tf.placeholder(tf.float32, [None, no_labels])

        y = ffn(x, no_hidden)

        # Create the model

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
        error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))
        train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        # train
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(no_iters):
                train.run(feed_dict={x:x_train, y_: y_train})
            err_ = error.eval(feed_dict={x:x_test, y_:y_test})
            err.append(err_)

#    print((np.argmin(err)+1)*2)
    return err


def main():
    
    # input data
    iris = datasets.load_iris()
    iris.data -= np.mean(iris.data, axis=0)
    n = iris.data.shape[0]

    print(iris.target.shape)

    X = iris.data
    no_data = len(iris.data)
    Y = np.zeros((no_data, no_labels))
    for i in range(no_data):
        Y[i, iris.target[i]] = 1
        

    err = []
    for exp in range(no_exps):
        print('exp %d'%exp)
        
        idx = np.arange(n)
        np.random.shuffle(idx)

        err.append(train_exp(X[idx], Y[idx]))

    mean_err = np.mean(np.array(err), axis = 0)
    print(mean_err)

    print(' hidden units %d '%hidden_units[np.argmin(mean_err)])
    
    plt.figure(1)
    plt.plot(hidden_units, mean_err, marker = 'x', linestyle = 'None')
    plt.xlabel('number of hidden units')
    plt.ylabel('mean classification error')
    plt.savefig('./figures/6.1b_1.png')

    plt.show()

if __name__ == '__main__':
    main()



        

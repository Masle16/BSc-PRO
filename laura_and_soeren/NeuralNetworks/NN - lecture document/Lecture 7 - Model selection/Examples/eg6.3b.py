#
# Chapter 6, example 3b
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
no_folds = 3
hidden_units = [2, 4, 6, 8, 10, 12]

seed = 10

tf.set_random_seed(seed)
np.random.seed(seed)

def ffn(x, no_hidden):

  # Hidden 
  with tf.name_scope('hidden'):
    weights = tf.Variable(
      tf.truncated_normal([no_features, no_hidden],
                            stddev=1.0 / np.sqrt(float(no_features))),
        name='weights')
    biases = tf.Variable(tf.zeros([no_hidden]),
                         name='biases')
    hidden = tf.nn.relu(tf.matmul(x, weights) + biases)
    
  # output
  with tf.name_scope('linear'):
    weights = tf.Variable(
        tf.truncated_normal([no_hidden, no_labels],
                            stddev=1.0 / np.sqrt(float(no_hidden))),
        name='weights')
    biases = tf.Variable(tf.zeros([no_labels]),
                         name='biases')
    logits = tf.matmul(hidden, weights) + biases
    
  return logits


def train_exp(X, Y):

    x_test, y_test = X[:30], Y[:30]
    XX, YY = X[30:], Y[30:]

    mean_err = []
    for no_hidden in hidden_units:

        x = tf.placeholder(tf.float32, [None, no_features])
        y_ = tf.placeholder(tf.float32, [None, no_labels])

        y = ffn(x, no_hidden)

        # Create the model
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

        error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))

        train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)


        
        err = []
        for fold in range(no_folds):
            start, end = fold*40, (fold+1)*40
            x_valid, y_valid = XX[start:end], YY[start:end]
            x_train  = np.append(XX[:start], XX[end:], axis=0)
            y_train = np.append(YY[:start], YY[end:], axis=0) 
            
            # train
            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                for i in range(no_iters):
                    train.run(feed_dict={x:x_train, y_: y_train})
                    
                err.append(error.eval(feed_dict={x:x_valid, y_:y_valid}))

        mean_err.append(np.mean(err))

        print('hidden units %d mean error = %g'% (no_hidden, np.mean(err)))

    no_hidden = hidden_units[np.argmin(mean_err)]
    

    x_train, y_train = XX, YY

    x = tf.placeholder(tf.float32, [None, no_features])
    y_ = tf.placeholder(tf.float32, [None, no_labels])

    y = ffn(x, no_hidden)

    # Create the model
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

    error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))

    train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(no_iters):
            train.run(feed_dict={x:x_train, y_: y_train})
        err = error.eval(feed_dict={x:x_test, y_:y_test})

    return no_hidden, err


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

    no_hidden, err = [], []
    for exp in range(no_exps):

        idx = np.arange(n)
        np.random.shuffle(idx)
        
        no_hidden_, err_ = train_exp(X[idx], Y[idx])
        
        no_hidden.append(no_hidden_)
        err.append(err_)
        print('exp %d hidden units %d error %g'%(exp, no_hidden[exp], err[exp]))

    hidden_counts = np.zeros(len(hidden_units))
    for exp in range(no_exps):
        hidden_counts[no_hidden[exp]//2-1] += 1

    print(hidden_counts)
    print('* %d *'%hidden_units[np.argmax(hidden_counts)])

    print('error = %g'%np.mean(err))
             

    plt.figure(1)
    plt.plot(range(no_exps), no_hidden, marker = 'x', linestyle = 'None')
    plt.yticks(hidden_units)
    plt.xticks(range(no_exps), np.arange(no_exps)+1)
    plt.xlabel('experiment')
    plt.ylabel('optimum number of hidden units')
 
    plt.savefig('./figures/6.3b_1.png')


    plt.figure(2)
    plt.plot(range(no_exps), err, marker = 'x', linestyle = 'None')
    plt.xticks(range(no_exps), np.arange(no_exps)+1)
    plt.xlabel('experiment')
    plt.ylabel('test error')
 
    plt.savefig('./figures/6.3b_2.png')
    
    plt.show()


if __name__ == '__main__':
    main()



        

#
# Tutorial 6, Question 2
#

from sklearn import datasets
import numpy as np
import tensorflow as tf
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


no_iters = 2500
no_features = 2
no_exps = 10
no_folds = 5

hidden_units = [2, 4, 6, 8, 10]

seed = 10

tf.set_random_seed(seed)
np.random.seed(seed)


# build a feedforward network
def ffn(x, hidden_units):

  # Hidden 
  with tf.name_scope('hidden'):
    weights = tf.Variable(
      tf.truncated_normal([no_features, hidden_units],
                            stddev=1.0 / np.sqrt(float(no_features))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),
                         name='biases')
    h = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
    
  # output
  with tf.name_scope('linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units, 1],
                            stddev=1.0 / np.sqrt(float(hidden_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([1]),
                         name='biases')
    u = tf.matmul(h, weights) + biases
    
  return u


# train the network and find errors
def train_exp(X, Y):

    err = []
    for fold in range(no_folds):
        start, end = fold*20, (fold+1)*20
        x_test, y_test = X[start:end], Y[start:end]
        x_train  = np.append(X[:start], X[end:], axis=0)
        y_train = np.append(Y[:start], Y[end:], axis=0) 

        err_ = []
        for no_hidden in hidden_units:
            x = tf.placeholder(tf.float32, [None, no_features])
            y_ = tf.placeholder(tf.float32, [None, 1])

            y = ffn(x, no_hidden)

            # Create the model

            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))
        
            train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        
            # train
            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                for i in range(no_iters):
                    train.run(feed_dict={x:x_train, y_: y_train})
                    
                err_.append(loss.eval(feed_dict={x:x_test, y_:y_test}))
        
        err.append(err_)

    
    cv_err = np.mean(np.array(err), axis = 0)
    print('cv errors {}'.format(cv_err))

    return cv_err


def main():

    # generate training data
    X = np.zeros((10*10, no_features))

    no_data = 0
    for i in np.arange(-1.0, 1.001, 2.0/9.0):
        for j in np.arange(-1.0, 1.001, 2.0/9.0):
            X[no_data] = [i, j]
            no_data += 1
    Y = np.zeros((no_data, 1))
    Y[:,0] = np.sin(np.pi*X[:,0])*np.cos(2*np.pi*X[:,1])

    # perform experiments
    idx = np.arange(no_data)
    err = []
    for exp in range(no_exps):
        print('exp %d'%exp)
 
        np.random.shuffle(idx)
        err.append(train_exp(X[idx], Y[idx]))

    # compute mean error
    mean_err = np.mean(np.array(err), axis = 0)
    print(mean_err)

    print(' hidden units %d '%hidden_units[np.argmin(mean_err)])
    
    plt.figure(1)
    plt.plot(hidden_units, mean_err, marker = 'x', linestyle = 'None')
    plt.xticks(hidden_units)
    plt.xlabel('number of hidden units')
    plt.ylabel('mean error')
    plt.savefig('./figures/t6q2_1.png')

    plt.show()


if __name__ == '__main__':
    main()



        

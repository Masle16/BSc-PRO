#
# Tutorial 6, Question 3
#

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
    
  with tf.name_scope('hidden'):
    weights = tf.Variable(
      tf.truncated_normal([no_features, hidden_units],
                            stddev=1.0 / np.sqrt(float(no_features))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),
                         name='biases')
    h = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
    
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

    x_test, y_test = X[:34], Y[:34]
    x_train, y_train = X[34:77], Y[34:77]
    x_valid, y_valid = X[77:], Y[77:]

    err = []
    for no_hidden in hidden_units:

        # create the model
        x = tf.placeholder(tf.float32, [None, no_features])
        y_ = tf.placeholder(tf.float32, [None, 1])

        y = ffn(x, no_hidden)

        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))
        train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        
        # train the model
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(no_iters):
                train.run(feed_dict={x:x_train, y_: y_train})
                    
            err.append(loss.eval(feed_dict={x:x_valid, y_:y_valid}))
        
    no_hidden = hidden_units[np.argmin(err)]       

   
    x_train, y_train = X[34:], Y[34:]

    # create the optimal model
    x = tf.placeholder(tf.float32, [None, no_features])
    y_ = tf.placeholder(tf.float32, [None, 1])

    y = ffn(x, no_hidden)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))
    train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # train the optimal model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(no_iters):
            train.run(feed_dict={x:x_train, y_: y_train})
        err = loss.eval(feed_dict={x:x_test, y_:y_test})

    return no_hidden, err



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

    idx = np.arange(no_data)

    # perform experiments
    no_hidden, err = [], []
    for exp in range(no_exps):

        np.random.shuffle(idx)
        no_hidden_, err_ = train_exp(X[idx], Y[idx])
        
        no_hidden.append(no_hidden_)
        err.append(err_)
        print('exp %d hidden units %d error %g'%(exp, no_hidden[exp], err[exp]))

    # find the consensus of experiments
    hidden_counts = np.zeros(len(hidden_units))
    for exp in range(no_exps):
        hidden_counts[no_hidden[exp]//2-1] += 1

    print(hidden_counts)
    print('* %d *'%hidden_units[np.argmax(hidden_counts)])
    print('error = %g'%np.mean(err))
             

    # plot the results
    plt.figure(1)
    plt.plot(range(no_exps), no_hidden, marker = 'x', linestyle = 'None')
    plt.yticks(hidden_units)
    plt.xticks(range(no_exps), np.arange(no_exps)+1)
    plt.xlabel('experiment')
    plt.ylabel('optimum number of hidden units')
    plt.savefig('./figures/t6q3_1.png')


    plt.figure(2)
    plt.plot(range(no_exps), err, marker = 'x', linestyle = 'None')
    plt.xticks(range(no_exps), np.arange(no_exps)+1)
    plt.xlabel('experiment')
    plt.ylabel('test error')
    plt.savefig('./figures/t6q3_2.png')
    
    plt.show()



if __name__ == '__main__':
    main()



        

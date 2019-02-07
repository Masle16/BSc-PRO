#
# Project 1, part A, question 2
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import pylab as plt
#import multiprocessing as mp
import time

import os
if not os.path.isdir('Figures_A2'):
    print('creating the figures folder')
    os.makedirs('Figures_A2')

# Scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

## Define the constants
NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 2000
batch_sizes = [4, 8, 16, 32, 64]
beta = 10**(-6)
hidden_units = 10

seed = 10
np.random.seed(seed)


def train_network(X, Y, Xt, Yt, bs):
            # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
        
        ## Build the graph for the deep net
            # Hidden perceptron layer
        w1 = tf.Variable(
            tf.truncated_normal([NUM_FEATURES, hidden_units],
                                stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
            name='weights')
        b1 = tf.Variable(tf.zeros([hidden_units]),name='biases')
        z1 = tf.matmul(x, w1) + b1
        h1 = tf.nn.sigmoid(z1)

            # Output softmax layer
        w2 = tf.Variable(
            tf.truncated_normal([hidden_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden_units))),
            name='weights')
        b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits  = tf.matmul(h1, w2) + b2

            # Loss with L2 regularization
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=logits) )
        regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
        loss = cross_entropy + beta*regularization

            # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        te_acc, tr_err, time_ = [],[],[] # Container for accuracies

        ## Start session and train network
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            idx = np.arange(X.shape[0])

            for i in range(epochs):
                tic = time.time()
                    # Shuffle the data for mini batch
                np.random.shuffle(idx)
                X, Y = X[idx], Y[idx]
                
                for p in range(X.shape[0]// bs):
                        train_op.run(feed_dict={x: X[p*bs: (p+1)*bs],
                                                y_: Y[p*bs: (p+1)*bs]})

                toc = time.time()

                time_.append( toc-tic )
                tr_err.append(cross_entropy.eval(feed_dict={x: X, y_: Y}))
                te_acc.append( accuracy.eval(feed_dict={x: Xt, y_: Yt}))

                if i % 100 == 0:
                        print('iter %d: train error: %g test accuracy: %g batch size: %g' % (i, tr_err[i], te_acc[i], bs))

        return np.vstack((tr_err, te_acc, time_))


def main():
                          
    ## Import train data
    train_input = np.loadtxt('sat_train.txt',delimiter=' ')
    trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)

        # Experiment with small datasets
    #trainX = trainX[:100]
    #train_Y = train_Y[:100]
        
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0)) #scaling of input
    train_Y[train_Y == 7] = 6

    trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix, targets represented as so

    ## Import test data
    test_input = np.loadtxt('sat_test.txt',delimiter=' ')
    testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
    testX = scale(testX, np.min(trainX, axis=0), np.max(trainX, axis=0)) #scaling of test input with train data
    test_Y[test_Y == 7] = 6

    testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
    testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix

    time_ = []


    i = 1
    for bs in batch_sizes:
        err_ = train_network(trainX, trainY, testX, testY, bs)
        time_.append(  np.mean( err_[2]) )
        #te_acc.append(err_[1])
        
            # Plot learning curves
        plt.figure(i)
        plt.plot(range(epochs), err_[0])
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Training error')
        plt.title('Training error vs epochs for batch size %g' %bs)
        plt.savefig('Figures_A2/trainErr%d.png' %(bs))

        plt.figure(i+1)
        plt.plot(range(epochs), err_[1])
        plt.xlabel(str(epochs) + 'iterations')
        plt.ylabel('Test accuracy')
        plt.title('Test accuracy vs epochs for batch size %g' %bs)
        plt.savefig('Figures_A2/testAcc%d.png' %(bs))
        i += 2


    plt.figure(i)
    plt.scatter( batch_sizes, time_ )
    plt.xlabel('Batch size')
    plt.ylabel('Average time in ms per epoch')
    plt.title('Average time in ms for one epoch vs Batch size')
    plt.savefig('Figures_A2/Time.png')
    
    
    #plt.show()

if __name__== '__main__':
    main()

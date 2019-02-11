#
# Chapter 6, example 4b
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import pylab as plt
import multiprocessing as mp

from tensorflow.examples.tutorials.mnist import input_data

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

seed = 10
learning_rate = 0.01
no_epochs = 2000
batch_size = 64

hidden1_units = 625
hidden2_units = 100

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(seed)
np.random.seed(seed)

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


def train(para):
  # Import data

  mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
  trainX, testX  = mnist.train.images[:600], mnist.test.images[:100]
  trainY, testY = mnist.train.labels[:600], mnist.test.labels[:100]

  # Create the model
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  beta = tf.placeholder(tf.float32)

  # Build the graph for the deep net
  w1 = tf.Variable(
      tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                          stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
      name='weights')
  b1 = tf.Variable(tf.zeros([hidden1_units]),name='biases')
  h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

  w2 = tf.Variable(
      tf.truncated_normal([hidden1_units, hidden2_units],
                          stddev=1.0 / math.sqrt(float(hidden1_units))),
      name='weights')
  b2 = tf.Variable(tf.zeros([hidden2_units]), name='biases')
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

  w3 = tf.Variable(
      tf.truncated_normal([hidden2_units, NUM_CLASSES],
                          stddev=1.0 / math.sqrt(float(hidden2_units))),
      name='weights')
  b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
  y = tf.matmul(h2, w3) + b3

 

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=y_, logits=y)

  regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

  loss = tf.reduce_mean(cross_entropy + beta*regularization)

  # Add a scalar summary for the snapshot loss.
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

  N = len(trainX)
  idx = np.arange(N)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    test_acc = []
    for i in range(no_epochs):
        np.random.shuffle(idx)
        trainXX = trainX[idx]
        trainYY = trainY[idx]

        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
            train_op.run(feed_dict={x: trainXX[start:end], y_: trainYY[start:end], beta: para})
      
        test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
        if i%100 == 0:
            print('rate %g: iter %d, test accuracy %g'%(para, i,  test_acc[i]))

  return test_acc


def main():
    no_threads = mp.cpu_count()

    rates = [0.05, 0.01, 0.001, 0.0]

    p = mp.Pool(processes = no_threads)
    acc = p.map(train, rates)

    plt.figure()
    for i in range(len(rates)):
        plt.plot(range(no_epochs), acc[i], label='beta = {}'.format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.savefig('./figures/6.4b_1.png')
    plt.show()

if __name__ == '__main__':
  main()

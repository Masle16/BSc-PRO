#
# Chapter 5, example 5a
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import pylab as plt

from tensorflow.examples.tutorials.mnist import input_data

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

seed = 10
learning_rate = 0.01
no_iters = 20000
batch_size = 64

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(seed)
np.random.seed(seed)

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

def ffn(x, hidden1_units, hidden2_units):

  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
    
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
    
  return logits

def main():
  # Import data
  mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
  trainX, trainY  = mnist.train.images, mnist.train.labels
  testX, testY = mnist.test.images, mnist.test.labels

  # Create the model
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  # Build the graph for the deep net
  y = ffn(x, 625, 100)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(cross_entropy)

  # Add a scalar summary for the snapshot loss.
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(cross_entropy, global_step=global_step)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    test_acc = []
    for i in range(no_iters):
      batch = mnist.train.next_batch(batch_size)
      train_op.run(feed_dict={x: batch[0], y_: batch[1]})
      if i % 100 == 0:
        acc = accuracy.eval(feed_dict=
                            { x: mnist.test.images, y_: mnist.test.labels})
        test_acc.append(acc)
        print('iter %d: accuracy %g'%(i, acc))


  plt.figure(1)
  plt.plot(range(no_iters//100), test_acc)
  plt.xlabel('100 iterations')
  plt.ylabel('test accuracy')
  plt.savefig('./figures/5.5a_1.png')

  plt.show()

if __name__ == '__main__':
  main()

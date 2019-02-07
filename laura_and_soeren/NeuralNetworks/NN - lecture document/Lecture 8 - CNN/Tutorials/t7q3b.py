#
# Tutorial 7, Question 3b
#


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import pylab

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
FLAGS = None

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

no_epochs = 500
batch_size = 128

alpha = 1e-3
beta = 1e-6
gamma = 0.5

def cnn(x):

  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 25 feature maps.
  W_conv = weight_variable([9, 9, 1, 25])
  b_conv = bias_variable([25])
  u_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID') + b_conv
  h_conv = tf.nn.sigmoid(u_conv)

  # Pooling layer - downsamples by 4X.
  h_pool = tf.nn.max_pool(h_conv, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1], padding='VALID')
  

   # Fully connected layer 1
  W_fc = weight_variable([5 * 5 * 25, 10])
  b_fc = bias_variable([10])

  h_pool_flat = tf.reshape(h_pool, [-1, 5*5*25])

  y_conv = tf.matmul(h_pool_flat, W_fc) + b_fc

  return W_conv, W_fc, y_conv


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main():
  # Import data
  mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
  trainX, trainY  = mnist.train.images[:12000], mnist.train.labels[:12000]
  testX, testY = mnist.test.images[:2000], mnist.test.labels[:2000]

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  W_conv, W_fc, y_conv = cnn(x)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  regularization = tf.nn.l2_loss(W_conv) + tf.nn.l2_loss(W_fc)

  train_step1 = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)
  train_step2 = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy + beta*regularization)
  train_step3 = tf.train.MomentumOptimizer(alpha, gamma).minimize(cross_entropy)
  train_step4 = tf.train.MomentumOptimizer(alpha, gamma).minimize(cross_entropy + beta*regularization)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  N = len(trainX)
  idx = np.arange(N)
  with tf.Session() as sess:
    pylab.figure()

    print('gd ...')
    sess.run(tf.global_variables_initializer())

    test_acc = []
    for i in range(no_epochs):
      np.random.shuffle(idx)
      trainX, trainY = trainX[idx], trainY[idx]

      for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_step1.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
      
      test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
      print('iter %d: test accuracy %g'%(i, test_acc[i]))

    pylab.plot(np.arange(no_epochs), test_acc, label='gd')


    print('gd with decay...')
    sess.run(tf.global_variables_initializer())

    test_acc = []
    for i in range(no_epochs):
      np.random.shuffle(idx)
      trainX, trainY = trainX[idx], trainY[idx]

      for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_step2.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
      
      test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
      print('iter %d: test accuracy %g'%(i, test_acc[i]))

    pylab.plot(np.arange(no_epochs), test_acc, label='gd + decay')

    print('momentum ...')
    sess.run(tf.global_variables_initializer())

    test_acc = []
    for i in range(no_epochs):
      np.random.shuffle(idx)
      trainX, trainY = trainX[idx], trainY[idx]

      for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_step3.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
      
      test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
      print('iter %d: test accuracy %g'%(i, test_acc[i]))

    pylab.plot(np.arange(no_epochs), test_acc, label='gd + momentum')

    print('gd with decay and momentum ...')
    sess.run(tf.global_variables_initializer())

    test_acc = []
    for i in range(no_epochs):
      np.random.shuffle(idx)
      trainX, trainY = trainX[idx], trainY[idx]

      for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
          train_step4.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
      
      test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
      print('iter %d: test accuracy %g'%(i, test_acc[i]))

    pylab.plot(np.arange(no_epochs), test_acc, label='gd + decay + momentum')

    pylab.xlabel('epochs')
    pylab.ylabel('test accuracy')
    pylab.legend(loc='lower right')
    pylab.savefig('./figures/t7q3b_1.png')
    
    pylab.show()


if __name__ == '__main__':
  main()

#
# Chapter 10, Example 2a: multiplicative noise
#


import tensorflow as tf
import numpy as np
import pylab
from tensorflow.examples.tutorials.mnist import input_data

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(10)

noise_prob=0.1
no_epochs = 50
lr = 0.1
batch_size = 128

n_in = 28*28
n_hidden = 500

mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
trainX = mnist.train.images
testX = mnist.test.images

x = tf.placeholder(tf.float32, [None, n_in])

W = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=1/np.sqrt(n_in)))
b = tf.Variable(tf.zeros([n_hidden]))
b_prime = tf.Variable(tf.zeros([n_in]))
W_prime = tf.transpose(W)

noise = tf.distributions.Bernoulli(probs = 1-noise_prob).sample(tf.shape(x))
tilde_x = x*tf.cast(noise, tf.float32)
h = tf.sigmoid(tf.matmul(tilde_x, W) + b)
y = tf.sigmoid(tf.matmul(h, W_prime) + b_prime)

entropy = - tf.reduce_mean(tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), axis=1))

train_op = tf.train.GradientDescentOptimizer(lr).minimize(entropy)


idx = np.arange(len(trainX))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost = []
    for e in range(no_epochs):
        np.random.shuffle(idx)
        trainX = trainX[idx]

        cost_ = []
        for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
            _, cost__ = sess.run([train_op, entropy], feed_dict={x: trainX[start:end]})
            cost_.append(cost__)

        cost.append(np.mean(cost_))
            
        if e%1 == 0:
            print('epoch %d: cost %g'%(e, cost[e]))
    w = sess.run(W)
    tilde_xx = sess.run(tilde_x, {x: testX[:50]})
    yy = sess.run(y, {x: testX[:50]})
    
pylab.figure()
pylab.plot(range(no_epochs), cost)
pylab.xlabel('epochs')
pylab.ylabel('entropy')
pylab.savefig('./figures/10.2a_1.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w[:,i].reshape(28,28))
pylab.savefig('./figures/10.2a_2.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(trainX[i,:].reshape(28,28))
pylab.savefig('./figures/10.2a_3.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(tilde_xx[i,:].reshape(28,28))
pylab.savefig('./figures/10.2a_4.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(testX[i,:].reshape(28,28))
pylab.savefig('./figures/10.2a_5.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(yy[i,:].reshape(28,28))
pylab.savefig('figure_10.2a_6.png')

pylab.show()

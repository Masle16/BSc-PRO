#
# Chapter 10, Example 3c: sparse AE
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
tf.set_random_seed(seed)

noise_prob=0.1
no_epochs = 50
lr = 0.1
batch_size = 128

beta = 0.5
rho = 0.05

n_in = 28*28
n_hidden = 900

mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
trainX = mnist.train.images
testX = mnist.test.images

x = tf.placeholder(tf.float32, [None, n_in])

W = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=1/np.sqrt(n_in)))
b = tf.Variable(tf.zeros([n_hidden]))
b_prime = tf.Variable(tf.zeros([n_in]))
W_prime = tf.transpose(W)

h = tf.sigmoid(tf.matmul(x, W) + b)
y = tf.sigmoid(tf.matmul(h, W_prime) + b_prime)

mse = tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=1))
divergence = tf.reduce_sum(rho*tf.log(rho/tf.reduce_mean(h, axis=0)) +
                           (1 - rho)*tf.log((1 - rho)/(1-tf.reduce_mean(h, axis=0))))
loss = mse + beta*divergence

train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)


idx = np.arange(len(trainX))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost = []
    for e in range(no_epochs):
        np.random.shuffle(idx)
        trainX = trainX[idx]
        
        cost_ = []
        for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
            _, cost__ = sess.run([train_op, loss], feed_dict={x: trainX[start:end]})
            cost_.append(cost__)

        cost.append(np.mean(cost_))
            
        if e%1 == 0:
            print('epoch %d: cost %g'%(e, cost[e]))

    w = sess.run(W)
    h_, y_ = sess.run([h, y], {x: testX[:49]})
    

pylab.figure()
pylab.plot(range(no_epochs), cost)
pylab.xlabel('epochs')
pylab.ylabel('m.s.e')
pylab.savefig('figures/10.3c_1.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w[:,i].reshape(28,28))
pylab.savefig('figures/10.3c_2.png')


pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(testX[i,:].reshape(28,28))
pylab.savefig('figures/10.3c_3.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(y_[i,:].reshape(28,28))
pylab.savefig('figures/10.3c_4.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(h_[i,:].reshape(30,30))
pylab.savefig('figures/10.3c_5.png')

pylab.show()

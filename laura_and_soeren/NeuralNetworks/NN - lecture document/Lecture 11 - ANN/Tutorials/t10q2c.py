#
# Tutorial 10, Question 2c: sparse AE
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
np.random.rand(seed)
tf.set_random_seed(seed)

noise_prob=0.1
no_epochs = 1000
lr = 0.2
batch_size = 32

n_in = 100
n_hidden = 144

rho = 0.05
beta = 0.5

X = np.random.rand(100, n_in)

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


idx = np.arange(len(X))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost = []
    for e in range(no_epochs):
        np.random.shuffle(idx)
        X = X[idx]
        
        cost_ = []
        for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X), batch_size)):
            _, cost__ = sess.run([train_op, loss], feed_dict={x: X[start:end]})
            cost_.append(cost__)

        cost.append(np.mean(cost_))
            
        if e%100 == 0:
            print('epoch %d: cost %g'%(e, cost[e]))

    w = sess.run(W)
    h_, y_ = sess.run([h, y], {x: X})
    

pylab.figure()
pylab.plot(range(no_epochs), cost)
pylab.xlabel('epochs')
pylab.ylabel('loss')
pylab.savefig('figures/t10q2c_1.png')

pylab.figure()
pylab.gray()
for i in range(144):
    pylab.subplot(12, 12, i+1); pylab.axis('off'); pylab.imshow(w[:,i].reshape(10,10))
pylab.savefig('figures/t10q2c_2.png')


pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(X[i,:].reshape(10,10))
pylab.savefig('figures/t10q2c_3.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(y_[i,:].reshape(10,10))
pylab.savefig('figures/t10q2c_4.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(h_[i,:].reshape(12,12))
pylab.savefig('figures/t10q2c_5.png')

pylab.show()

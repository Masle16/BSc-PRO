#
# Tutorial 10, Question 3: deep AE classifier
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
np.random.seed(seed)
tf.set_random_seed(seed)

# noise_prob=0.1
no_epochs = 50
# lr = 0.1
batch_size = 128

beta = 0.4
rho = 0.02

n_in = 28*28
n_hidden1 = 625
n_hidden2 = 100
n_out = 10

mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
trainX, trainY = mnist.train.images, mnist.train.labels
testX, testY = mnist.test.images, mnist.test.labels

x = tf.placeholder(tf.float32, [None, n_in])
y_ = tf.placeholder(tf.float32, [None, 10])
noise_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([n_in, n_hidden1], stddev=1/np.sqrt(n_in)))
b1 = tf.Variable(tf.zeros([n_hidden1]))
b1_prime = tf.Variable(tf.zeros([n_in]))
W1_prime = tf.transpose(W1)

noise = tf.distributions.Bernoulli(probs = 1-noise_prob).sample(tf.shape(x))
tilde_x = x*tf.cast(noise, tf.float32)

h1 = tf.sigmoid(tf.matmul(tilde_x, W1) + b1)
y1 = tf.sigmoid(tf.matmul(h1, W1_prime) + b1_prime)

mse1 = tf.reduce_mean(tf.reduce_sum(tf.square(x - y1), axis=1))
divergence1 = tf.reduce_sum(rho*tf.log(rho/tf.reduce_mean(h1, axis=0)) +
                           (1 - rho)*tf.log((1 - rho)/(1-tf.reduce_mean(h1, axis=0))))
loss1 = mse1 + beta*divergence1

grad_W1, grad_b1, grad_b1_prime = tf.gradients(loss1, [W1, b1, b1_prime])
W1_new = W1.assign(W1 - lr*grad_W1)
b1_new = b1.assign(b1 - lr*grad_b1)
b1_prime_new = b1_prime.assign(b1_prime - lr*grad_b1_prime)

W2 = tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=1/np.sqrt(n_hidden1)))
b2 = tf.Variable(tf.zeros([n_hidden2]))
b2_prime = tf.Variable(tf.zeros([n_hidden1]))
W2_prime = tf.transpose(W2)

h2 = tf.sigmoid(tf.matmul(h1, W2) + b2)
y2 = tf.sigmoid(tf.matmul(h2, W2_prime) + b2_prime)

mse2 = tf.reduce_mean(tf.reduce_sum(tf.square(h1 - y2), axis=1))
divergence2 = tf.reduce_sum(rho*tf.log(rho/tf.reduce_mean(h2, axis=0)) +
                           (1 - rho)*tf.log((1 - rho)/(1-tf.reduce_mean(h2, axis=0))))
loss2 = mse2 + beta*divergence2

grad_W2, grad_b2, grad_b2_prime = tf.gradients(loss2, [W2, b2, b2_prime])
W2_new = W2.assign(W2 - lr*grad_W2)
b2_new = b2.assign(b2 - lr*grad_b2)
b2_prime_new = b2_prime.assign(b2_prime - lr*grad_b2_prime)

W3 = tf.Variable(tf.truncated_normal([n_hidden2, n_out], stddev=1/np.sqrt(n_hidden2)))
b3 = tf.Variable(tf.zeros([n_out]))

y3 = tf.matmul(h2, W3) + b3

loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y3))
train_op3 = tf.train.AdamOptimizer(lr).minimize(loss3)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y3, 1), tf.argmax(y_, 1)), tf.float32))

idx = np.arange(len(trainX))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('training 1st hidden-layer ...')
    cost1 = []
    for e in range(no_epochs):
        np.random.shuffle(idx)
        X = trainX[idx]
        
        cost_ = []
        for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X), batch_size)):
            _, _, _, cost__ = sess.run([W1_new, b1_new, b1_prime_new, loss1],
                                       feed_dict={x: X[start:end], noise_prob: 0.1, lr: 0.1})
            cost_.append(cost__)

        cost1.append(np.mean(cost_))
            
        if e%1 == 0:
            print('epoch %d: cost %g'%(e, cost1[e]))

    w1 = sess.run(W1)
    h1_, y1_ = sess.run([h1, y1], {x: testX[:49], noise_prob: 0.0})

    print('training 2nd hidden-layer ...')
    cost2 = []
    for e in range(no_epochs):
        np.random.shuffle(idx)
        X = trainX[idx]
        
        cost_ = []
        for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X), batch_size)):
            _, _, _, cost__ = sess.run([W2_new, b2_new, b2_prime_new, loss2],
                                       feed_dict={x: X[start:end], noise_prob: 0.1, lr: 0.1})
            cost_.append(cost__)

        cost2.append(np.mean(cost_))
            
        if e%1 == 0:
            print('epoch %d: cost %g'%(e, cost2[e]))

    w2 = sess.run(W2)
    h2_, y2_ = sess.run([h2, y2], {x: testX[:49], noise_prob: 0.0})

    print('training classifier ...')
    cost3, acc = [], []
    for e in range(no_epochs):
        np.random.shuffle(idx)
        X, Y = trainX[idx], trainY[idx]
        
        cost_, acc_ = [], []
        for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X), batch_size)):
            _, cost__, acc__ = sess.run([train_op3, loss3, accuracy],
                                 feed_dict={x: X[start:end], y_: Y[start:end], noise_prob: 0.0, lr: 0.001})
            cost_.append(cost__)
            acc_.append(acc__)

        cost3.append(np.mean(cost_))
        acc.append(np.mean(acc_))
            
        if e%1 == 0:
            print('epoch %d: cost %g'%(e, cost3[e]))

    w3 = sess.run(W3)
    h1__, h2__ = sess.run([h1, h2], {x: testX[:49], noise_prob: 0.0})


pylab.figure()
pylab.plot(range(no_epochs), cost1)
pylab.xlabel('epochs')
pylab.ylabel('cost')
pylab.savefig('figures/t10q3_1.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig('figures/t10q3_2.png')


pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(testX[i,:].reshape(28,28))
pylab.savefig('figures/t10q3_3.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(y1_[i,:].reshape(28,28))
pylab.savefig('figures/t10q3_4.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(h1_[i,:].reshape(25,25))
pylab.savefig('figures/t10q3_5.png')

pylab.figure()
pylab.plot(range(no_epochs), cost2)
pylab.xlabel('epochs')
pylab.ylabel('cost')
pylab.savefig('figures/t10q3_6.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(25,25))
pylab.savefig('figures/t10q3_7.png')


pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(h1_[i,:].reshape(25,25))
pylab.savefig('figures/t10q3_8.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(y2_[i,:].reshape(25,25))
pylab.savefig('figures/t10q3_9.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(h2_[i,:].reshape(10,10))
pylab.savefig('figures/t10q3_10.png')

pylab.figure()
pylab.plot(range(no_epochs), cost3)
pylab.xlabel('epochs')
pylab.ylabel('cost')
pylab.savefig('figures/t10q3_11.png')

pylab.figure()
pylab.plot(range(no_epochs), acc)
pylab.xlabel('epochs')
pylab.ylabel('accuracy')
pylab.savefig('figures/t10q3_12.png')

pylab.figure()
pylab.gray()
for i in range(10):
    pylab.subplot(2, 5, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(10,10))
pylab.savefig('figures/t10q3_13.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(h1__[i,:].reshape(25,25))
pylab.savefig('figures/t10q3_14.png')

pylab.figure()
pylab.gray()
for i in range(49):
    pylab.subplot(7, 7, i+1); pylab.axis('off'); pylab.imshow(h2__[i,:].reshape(10,10))
pylab.savefig('figures/t10q3_15.png')

pylab.show()

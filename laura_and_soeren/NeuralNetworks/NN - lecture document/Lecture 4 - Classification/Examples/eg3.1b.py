#
# Chapter 3, example 1b
#

import tensorflow as tf
import numpy as np
import pylab as plt
import multiprocessing as mp

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

no_iters = 30
SEED = 10
np.random.seed(SEED)

# training data
x_train = np.array([[1.0, 2.5], [2.0, -1.0], [1.5, 3.0],
	[0.0, -1.5], [-3.5, 1.0], [2.5, 0.0], [0.5, 1.5], [0.0, -2.0]])
y_train = np.array([1, 0, 1, 0, 1, 0, 0, 0])

# Model parameters
w = tf.Variable(np.random.rand(2), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)
lr = tf.Variable(0.4, dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
d = tf.placeholder(tf.int32)


u = tf.tensordot(x,w, axes=1) + b
y = tf.where(tf.greater(u, 0), 1, 0)
delta = d - y
delta = tf.cast(delta, tf.float32)

w_new = w.assign(w + lr*delta*x)
b_new = b.assign(b + lr*delta)


# training loop
def my_train(rate):
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init) # reset values to wrong

	X, Y = x_train, y_train
	err = []
	idx = np.arange(len(X))
	for i in range(no_iters):
		np.random.shuffle(idx)
		X, Y = X[idx], Y[idx]
		err_ = 0
		for p in np.arange(len(X)):
  			y_, w_, b_ = sess.run([y, w_new, b_new], {x: X[p], d: Y[p]})
  			err_ += y_ != Y[p]

		err.append(err_)

	return err


rates = [0.01, 0.05, 0.1, 0.5]


for i in range(len(rates)):
    cost = my_train(rates[i])
    plt.figure()
    plt.plot(range(no_iters), cost)
    plt.xlabel('epochs')
    plt.ylabel('classification error')
    plt.yticks([0, 1, 2, 3])
    plt.title('learning at {}'.format(rates[i]))
    plt.savefig('./figures/3.1b_{}.png'.format(rates[i]))


plt.show()



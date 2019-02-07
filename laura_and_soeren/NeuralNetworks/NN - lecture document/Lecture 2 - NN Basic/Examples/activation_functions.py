# 
# Activation functions
#

import numpy as np
import tensorflow as tf
import pylab


import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

def thresh(u):
	shape = tf.shape(u)
	return tf.where(tf.greater(u, tf.zeros(shape)), tf.ones(shape), tf.zeros(shape))

# input
U = np.arange(-10.0, 10.0, 0.1)

u = tf.placeholder(tf.float32)

sess = tf.Session()


# sigmoid
y = tf.sigmoid(u)
Y = sess.run(y, {u:U})

pylab.figure()
ax = pylab.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
pylab.xticks([-5, 5])
pylab.yticks([0.5, 1.0])
pylab.plot(U,Y)
pylab.xlabel('$u$')
pylab.title('sigmoid($u$)')
pylab.savefig('./figures/sigmoid.png')


# thresh
y = thresh(u)
Y = sess.run(y, {u:U})

pylab.figure()
ax = pylab.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
pylab.xticks([-5, 5])
pylab.yticks([0.5, 1.0])
pylab.plot(U,Y)
pylab.xlabel('$u$')
pylab.title('thresh($u$)')
pylab.savefig('./figures/thresh.png')


# linear
y = u
Y = sess.run(y, {u:U})

pylab.figure()
ax = pylab.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
pylab.xticks([-5, 5])
pylab.yticks([-5, 5])
pylab.plot(U,Y)
# pylab.xlabel('$u$')
pylab.title('linear($u$)')
pylab.savefig('./figures/linear.png')


# relu
y = tf.nn.relu(u)
Y = sess.run(y, {u:U})

pylab.figure()
ax = pylab.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
pylab.xticks([-5, 5])
pylab.yticks([5])
pylab.plot(U,Y)
pylab.xlabel('$u$')
pylab.title('ReLU($u$)')
pylab.savefig('./figures/relu.png')


# tanh
y = tf.tanh(u)
Y = sess.run(y, {u:U})

pylab.figure()
ax = pylab.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
pylab.xticks([-5,  5])
pylab.yticks([-0.5, 0.5])
pylab.plot(U,Y)
# pylab.xlabel('$u$')
pylab.title('tanh($u$)')
pylab.savefig('./figures/tanh.png')

pylab.show()


#
# Chapter 7, Example 1
#

import tensorflow as tf
import numpy as np
import pylab


# Input image
I = np.array([[0.7, 0.1, 0.2, 0.3, 0.3, 0.5],
              [0.8, 0.1, 0.3, 0.5, 0.1, 0.0],
              [1.0, 0.2, 0.0, 0.3, 0.2, 0.7],
              [0.8, 0.1, 0.5, 0.6, 0.3, 0.4],
              [0.1, 0.0, 0.9, 0.3, 0.3, 0.2],
              [1.0, 0.1, 0.4, 0.5, 0.2, 0.8]]).astype(np.float32)

# filter
W = np.array([[[0, 1, 1],[1, 0, 1], [1, 1, 0]],
              [[-1, -1, -1],[0, 0, 0], [1, 1, 1]]]).astype(np.float32)
W = W.transpose((1, 2, 0))
B = np.array([0.1, 0.1]).astype(np.float32)

# computational graph
x = tf.placeholder(tf.float32, [1, 6, 6, 1])

w = tf.Variable(W.reshape(3, 3, 1, 2), tf.float32)
b = tf.Variable(B, tf.float32)

u1 = tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'VALID') + b
u2 = tf.nn.conv2d(x, w, strides = [1, 2, 2, 1], padding = 'SAME') + b
y1 = tf.nn.sigmoid(u1)
y2 = tf.nn.sigmoid(u2)

# initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# evaluate u and y
u1_, y1_ = sess.run([u1, y1], {x: I.reshape([1, 6, 6, 1])})

print('VALID padding and strides = (1, 1) for convolution')
print('u: %s'%u1_.transpose((0, 3, 1, 2)))
print('y: %s'%y1_.transpose((0, 3, 1, 2)))

u2_, y2_ = sess.run([u2, y2], {x: I.reshape([1, 6, 6, 1])})

print('SAME padding and strides = (2, 2) for convolution')
print('u: %s'%u2_.transpose((0, 3, 1, 2)))
print('y: %s'%y2_.transpose((0, 3, 1, 2)))

# evaluate o for max pooling
print('MAX pooling')
o1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
o1_ = sess.run(o1, {x: I.reshape([1, 6, 6, 1])})
print('o: %s'%o1_.transpose((0, 3, 1, 2)))

o2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
o2_ = sess.run(o2, {x: I.reshape([1, 6, 6, 1])})
print('o: %s'%o2_.transpose((0, 3, 1, 2)))

# evaluate for avg pooling
print('AVG pooling')
o3 = tf.nn.avg_pool(y1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
o3_ = sess.run(o3, {x: I.reshape([1, 6, 6, 1])})
print('o: %s'%o3_.transpose((0, 3, 1, 2)))

o4 = tf.nn.avg_pool(y2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
o4_ = sess.run(o4, {x: I.reshape([1, 6, 6, 1])})
print('o: %s'%o4_.transpose((0, 3, 1, 2)))

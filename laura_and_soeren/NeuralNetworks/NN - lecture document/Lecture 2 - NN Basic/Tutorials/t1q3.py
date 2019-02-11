#
# Tutorial 1, Question 3
#

import tensorflow as tf
import numpy as np


# threshold activation function
def thresh(u):
    shape = tf.shape(u)
    return tf.where(tf.greater(u, tf.zeros(shape)), tf.ones(shape), tf.zeros(shape))


# build the graph   
x = tf.placeholder(tf.float32)

w1 = tf.Variable([1.0, 1.0, 1.0], tf.float32)
b1 = tf.Variable(1/2, tf.float32)
w2 = tf.Variable([1.0, 1.0, 1.0], tf.float32)
b2 = tf.Variable(3/2, tf.float32)
w3 = tf.Variable([1.0, 1.0, 1.0], tf.float32)
b3 = tf.Variable(5/2, tf.float32)
w4 = tf.Variable([1.0, -1.0, 1.0], tf.float32)
b4 = tf.Variable(1/2, tf.float32)

u1 = tf.tensordot(w1, x, axes=1) - b1
y1 = thresh(u1)
u2 = tf.tensordot(w2, x, axes=1) - b2
y2 = thresh(u2)
u3 = tf.tensordot(w3, x, axes=1) - b3
y3 = thresh(u3)
u = tf.tensordot(w4, [y1, y2, y3], axes=1) - b4
y = thresh(u)


# evaluate the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2):
    for j in range(2):
        for k in range(2):
            u1_, y1_, u2_, y2_, u3_, y3_, u_, y_ = sess.run(
                [u1, y1, u2, y2, u3, y3, u, y], {x: [i, j, k]})

            print('u1=%.1f, u2=%.1f, u3=%.1f; y1=%d, y2=%d, y3=%d, u=%.1f, y=%d'
                  %(u1_, u2_, u3_, y1_, y2_, y3_, u_, y_))
    

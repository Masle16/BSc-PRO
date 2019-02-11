#
# Tutorial 1, Question 1
#

import tensorflow as tf
import numpy as np


def g(u):
    return 1.0/(1.0 + tf.exp(-0.5*u))

def f(u):
    shape = tf.shape(u)
    return tf.where(tf.greater(u, tf.zeros(shape)), u, tf.zeros(shape))


# build the graph   
x = tf.placeholder(tf.float32)

w1 = tf.Variable([1.0, -0.5, -1.0], tf.float32)
b1 = tf.Variable(0.0, tf.float32)
w2 = tf.Variable([0.0, 2.0, 0.6], tf.float32)
b2 = tf.Variable(0.5, tf.float32)
w3 = tf.Variable([-0.5, 0.6], tf.float32)
b3 = tf.Variable(0.05, tf.float32)

u1 = tf.tensordot(w1, x, axes=1) + b1
y1 = g(u1)
u2 = tf.tensordot(w2, x, axes=1) + b2
y2 = g(u2)
u3 = tf.tensordot(w3, [y1, y2], axes=1) + b3
y3 = f(u3)


# evaluate the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

u1_, y1_, u2_, y2_, u3_, y3_ = sess.run([u1, y1, u2, y2, u3, y3],
                                        {x: [1.0, -0.5, 1.0]})
print('x = ',[1.0, -0.5, 1.0])
print('u1=%g, u2=%g, u3=%g'% (u1_, u2_, u3_))
print('y1=%g, y2=%g, y3=%g'%(y1_, y2_, y3_))

u1_, y1_, u2_, y2_, u3_, y3_ = sess.run([u1, y1, u2, y2, u3, y3],
                                        {x: [-1.0, 0.0, -2.0]})
print('\n x = ',[-1.0, 0.0, -2.0])
print('u1=%g, u2=%g, u3=%g'% (u1_, u2_, u3_))
print('y1=%g, y2=%g, y3=%g'%(y1_, y2_, y3_))

u1_, y1_, u2_, y2_, u3_, y3_ = sess.run([u1, y1, u2, y2, u3, y3],
                                        {x: [2.0, 0.5, -1.0]})
print('\n x = ',[2.0, 0.5, -1.0])
print('u1=%g, u2=%g, u3=%g'% (u1_, u2_, u3_))
print('y1=%g, y2=%g, y3=%g'%(y1_, y2_, y3_))

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
no_features = 2
no_classes = 3

SEED = 10
np.random.seed(SEED)

# data
X = np.array([[0.5, -1.66],[-1.0, -0.51],[0.78, -0.65],[0.04, -0.20]])

# Model parameters
w = tf.Variable(np.random.normal(0., 0.1, (no_features, no_classes)), dtype=tf.float32)
b = tf.Variable(0.1*np.random.rand(no_classes), dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, X.shape)


u = tf.matmul(x, w) + b
y = tf.sigmoid(u)

print('x:{}'.format(X))

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

u_, y_ = sess.run([u, y], {x: X})
print('u:{}'.format(u_))
print('p:{}'.format(y_))



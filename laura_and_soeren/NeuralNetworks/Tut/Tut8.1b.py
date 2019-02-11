# input 28x28, feature maps x2,
# 25  Filters of 9x9
# Polling window of 4x4 

import tensorflow as tf
import numpy as np
import pylab 



## Define parameters
n_in = 3
n_hidden = 2
n_out = 1
n_steps =4
n_seqs =1

BATCH_SIZE = 128
LR = 10**(-3)
EPOCHS = 500
SEED = 10

np.random.seed(SEED)
tf.set_random_seed(SEED)

trainX = np.array([[ 1,  2, -1], [-1, 1, -2], [ 0, 3, 1], [2, -1, 0]])
X      = np.array([[ 1,  2, -1], [-1, 1, -2], [ 0, 3, 1], [2, -1, 0]])

Ug = np.array([[-1.0,  0.5],[ 0.5,  0.1],[ 0.2, -2.0]])
Wg = np.array([[2.0, 1.3]])
bg = np.array([0.2,0.2])
Vg = np.array([[2.0], [-1.5]])
cg = [0.5]

# Build the graph
U = tf.Variable(Ug, dtype= tf.float32)
W = tf.Variable(Wg, dtype= tf.float32)
b = tf.Variable(bg, dtype= tf.float32)
V = tf.Variable(Vg, dtype= tf.float32)
c = tf.Variable(cg, dtype= tf.float32)

x = tf.placeholder(tf.float32, trainX.shape[1])
y_old = tf.placeholder(tf.float32, [n_out])


z = tf.tensordot( tf.transpose(U),x, axes=1) + tf.tensordot(tf.transpose(W), y_old,axes=1) + b
h = tf.tanh(z)
u = tf.tensordot(tf.transpose(V), h, axes=1) + c
y = tf.sigmoid(u)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

y_ = np.array([0])
for t in range(n_steps):
    h_, y_ = sess.run([h, y], {x: X[t], y_old: y_})
    print('h: {}'.format(h_))
    print('y: {}\n'.format(y_))



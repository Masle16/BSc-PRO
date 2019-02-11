#
# Tutorial 8, Question 1b
#


import numpy as np
import tensorflow as tf


n_hidden = 2
n_in = 3
n_out = 1
n_steps = 4
n_seq = 1


X = np.array([[1, 2, -1], [-1, 1, -2], [0, 3, 1], [2, -1, 0]])


W = tf.Variable(np.array([[2.0, 1.3]]), dtype=tf.float32)
b = tf.Variable(np.array([0.2, 0.2]), dtype=tf.float32)
U = tf.Variable(np.array([[-1.0, 0.5], [0.5, 0.1], [0.2, -2]]), dtype=tf.float32)
V = tf.Variable(np.array([[2.0], [-1.5]]), dtype=tf.float32)
c = tf.Variable(np.array([0.5]), dtype=tf.float32)

x = tf.placeholder(tf.float32, X.shape[1])
init_state = tf.placeholder(tf.float32, [n_out])


z = tf.tensordot(tf.transpose(U), x, axes=1) + tf.tensordot(tf.transpose(W), init_state, axes=1) + b
h = tf.tanh(z)
u = tf.tensordot(tf.transpose(V), h, axes=1) + c
y = tf.sigmoid(u)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


y_ = np.array([0])
for t in range(n_steps):
    h_, y_ = sess.run([h, y], {x: X[t], init_state: y_})
    print('h: {}'.format(h_))
    print('y: {}'.format(y_))

         
            



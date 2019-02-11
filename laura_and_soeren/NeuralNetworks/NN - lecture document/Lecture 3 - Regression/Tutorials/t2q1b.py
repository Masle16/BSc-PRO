#
# Tutorial 2, Question 1b
#


import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

	
no_iters = 200
lr = 0.01

SEED = 10
np.random.seed(SEED)

# generate training data
X = np.array([[0.09, -0.44, -0.15],
              [0.69, -0.99, -0.76],
              [0.34, 0.65, -0.73],
              [0.15, 0.78, -0.58],
              [-0.63, -0.78, -0.56],
              [0.96, 0.62, -0.66],
              [0.63, -0.45, -0.14],
              [0.88, 0.64, -0.33]])
Y = np.array([-2.57, -2.97, 0.96, 1.04, -3.21, 1.05, -2.39, 0.66])
Y = Y.reshape(8,1)

print('x: %s'%X)
print('d: %s'%Y)

# Model parameters
w = tf.Variable(np.random.rand(3,1), dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [None, 3])
d = tf.placeholder(tf.float32, [None, 1])

y = tf.matmul(x,w) + b
loss = tf.reduce_mean(tf.square(d - y)) # sum of the squares

# optimizer
grad_w = -tf.matmul(tf.transpose(x), d - y)
grad_b = -tf.reduce_sum(d - y)
w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) # intialize values 
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err = []
for i in range(no_iters):

  if i < 2 or i == no_iters - 1:
      y_, loss_ = sess.run([y, loss], {x: X, d: Y})
      print('iter: {}'.format(i+1))
      print('y: {}'.format(y_))
      print('mse: {}'.format(loss_))

  sess.run([w_new, b_new], {x: X, d: Y})
  err.append(sess.run(loss, {x: X, d: Y}))

  if (i < 2 or i == no_iters - 1):
      w_, b_ = sess.run([w, b], {x: X, d: Y})
      print('w: {}, b: {}'.format(w_, b_))

  if i%10 == 0:
      print('iter: %d, error: %g'%(i, err[i]))

plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.savefig('./figures/t2q1b_1.png')


plt.show()



#
# Tutorial 2, Example 1a
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

# Model parameters
w = tf.Variable(np.random.rand(3), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [3])
d = tf.placeholder(tf.float32)

y = tf.tensordot(x, w, axes=1) + b
loss = tf.square(d - y) # sum of the squares

# optimizer
grad_w = -(d - y)*x
grad_b = -(d - y)
w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

mse = []
idx = np.arange(len(X))
for i in range(no_iters):
 
  np.random.shuffle(idx)
  X, Y = X[idx], Y[idx]
  
  err_ = []
  for p in np.arange(len(X)):
    y_, loss_, w_, b_ = sess.run([y, loss, w_new, b_new], {x: X[p], d: Y[p]})

    if (i == 0):
      print('iter: %d'%(i+1))
      print('p: %d'%(p+1))
      print('x:{}, d:{}'.format(X[p], Y[p]))
      print('y: %g'%(y_))
      print('loss: %g'%(loss_))
      print('w: {}, b: {}'.format(w_, b_))

    err_.append(loss_)

  mse.append(np.mean(err_))
  if i%10 == 0:
          print('iter: %d, error: %g'%(i, mse[i]))

 
print('w: {}, b: {}'.format(w_, b_))
print('mse: %g'%mse[no_iters-1])

# plot learning curve
plt.figure(1)
plt.plot(range(no_iters), mse)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.savefig('./figures/t2q1a_1.png')

# print predictions
for p in np.arange(len(X)):
    y_ = sess.run(y, {x:X[p]})
    print('x: %s, d: %g, y: %g'%(X[p], Y[p], y_))

plt.show()






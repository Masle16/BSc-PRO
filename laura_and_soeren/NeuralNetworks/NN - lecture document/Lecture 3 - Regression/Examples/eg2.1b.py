#
# Chapter 2, Example 1: using tf.gradients
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')
	
no_iters = 200
lr = 0.01
SEED = 10
np.random.seed(SEED)

# generate training data
X = 2*np.random.rand(6, 2) - 1
Y = np.dot(X, [2.53, -0.47]) + np.random.rand(6) - 0.5

# Model parameters
w = tf.Variable(np.random.rand(2), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [2])
d = tf.placeholder(tf.float32)

y = tf.tensordot(x,w, axes=1) + b
loss = tf.reduce_sum(tf.square(d - y)) # sum of the squares

# optimizer
grad_w, grad_b = tf.gradients(loss, [w, b])
w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err = []
idx = np.arange(len(X))
for i in range(no_iters):
  err_ = []
  np.random.shuffle(idx)
  X, Y = X[idx], Y[idx]
  for p in np.arange(len(X)):
    y_, loss_, w_, b_ = sess.run([y, loss, w_new, b_new], {x: X[p], d: Y[p]})

    if (i == 0 or i == no_iters-1):
      print('iter: {}'.format(i+1))
      print('p: {}'.format(p+1))
      print('x:{}, d:{}'.format(X[p], Y[p]))
      print('y: {}'.format(y_))
      print('loss: {}'.format(loss_))
      print('w: {}, b: {}'.format(w_, b_))

    err_.append(loss_)
  err.append(np.mean(err_))
  if i%10 == 0:
          print('iter: {}, error: {}'.format(i, err[i]))

 
# evaluate training accuracy
print('w: {}, b: {}'.format(w_, b_))

plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.savefig('./figures/2.1b_1.png')

pred = []
for p in np.arange(len(X)):
	pred.append(sess.run(y, {x:X[p]}))

fig = plt.figure(2)
ax = fig.gca(projection = '3d')
plot_original = ax.scatter(X[:,0], X[:,1], Y, 'ro', label='targets')
plot_pred = ax.scatter(X[:,0], X[:,1], pred, 'b^', label='predicted')
X1 = np.arange(-1, 1, 0.1)
X2 = np.arange(-1, 1, 0.1)
X1,X2 = np.meshgrid(X1,X2)
Z = w_[0]*X1 + w_[1]*X2 + b_
regression_plane = ax.plot_surface(X1, X2, Z)
ax.set_zticks([ -2, -1, 0, 1])
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.title('targets and predictions')
plt.legend()
plt.savefig('./figures/2.1b_2.png')



plt.show()



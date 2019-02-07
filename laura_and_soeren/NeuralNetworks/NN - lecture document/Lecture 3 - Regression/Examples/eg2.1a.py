#
# Chapter 2, Example 1
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
Y = np.dot(X, [2.53, -0.47]) - 0.5 + np.random.rand(6)

print(X)
print(Y)
print(lr)

# Model parameters
w = tf.Variable(np.random.rand(2), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [2])
d = tf.placeholder(tf.float32)

y = tf.tensordot(x, w, axes=1) + b
loss = tf.square(d - y) # sum of the squares

# optimizer
grad_w = -(d - y)*x
grad_b = -(d - y)
w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)

# initialize variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# print initial weights and biases
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

# training loop begins
err = []
idx = np.arange(len(X))
for i in range(no_iters):
        
  err_ = []
  np.random.shuffle(idx)
  X, Y = X[idx], Y[idx]
  for p in np.arange(len(X)):
    y_, loss_, w_, b_ = sess.run([y, loss, w_new, b_new], {x: X[p], d: Y[p]})

    if i == 0:
      print('iter: {}'.format(i+1))
      print('p: {}'.format(p+1))
      print('x:{}, d:{}'.format(X[p], Y[p]))
      print('y: {}'.format(y_))
      print('se: {}'.format(loss_))
      print('w: {}, b: {}'.format(w_, b_))

    err_.append(loss_)
  err.append(np.mean(err_))
  if i%10 == 0:
          print('iter: %d, mse: %g'%(i, err[i]))

# print final weights and error
w_, b_ = sess.run([w, b])
print('w: %s, b: %s'%(w_, b_))
print('mse: %g'%err[no_iters-1])

# plot learning curve
plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.savefig('./figures/2.1a_1.png')

# find the predicted values of inputs
pred = []
for p in np.arange(len(X)):
	pred.append(sess.run(y, {x:X[p]}))

# plot targets and predictions
fig = plt.figure(2)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y, 'ro', label='targets')
ax.scatter(X[:,0], X[:,1], pred, 'b^', label='predicted')

X1 = np.arange(-1, 1, 0.1)
X2 = np.arange(-1, 1, 0.1)
X1,X2 = np.meshgrid(X1,X2)
Z = w_[0]*X1 + w_[1]*X2 + b_
ax.plot_surface(X1, X2, Z)

ax.set_zticks([ -2, -1, 0, 1])
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.title('targets and predictions')
plt.legend()
plt.savefig('./figures/2.1a_2.png')



plt.show()



#
# Tutorial 2, Question 2: Stochastic Gradient Descent
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

	
no_iters = 500
lr = 0.01

SEED = 10
np.random.seed(SEED)

# generate data
no_data = 11*11
X = np.zeros((no_data,2))
Y = np.zeros(no_data)
i = 0
for x1 in np.arange(0, 1.01, 0.1):
    for x2 in np.arange(0, 1.01, 0.1):
        X[i] = [x1, x2]
        Y[i] = 0.5+x1+3*x2**2
        i += 1


# Model parameters
w = tf.Variable(np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [2])
d = tf.placeholder(tf.float32)

u = tf.tensordot(x, w, axes=1) + b
y = 4*tf.sigmoid(u) + 0.5
loss = tf.square(d - y) # square error


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
  
  np.random.shuffle(idx)
  X, Y = X[idx], Y[idx]

  err_ = []
  for p in np.arange(len(X)):
    sess.run([w_new, b_new], {x: X[p], d: Y[p]})
    err_.append(sess.run(loss, {x: X[p], d: Y[p]}))

  err.append(np.mean(err_))
  
  if i%100 == 0:
     print('iter: %d, mse: %g'%(i, err[i]))

w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))
print('mse = %g'%err[no_iters-1])

# plot learning curves
plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.title('Training Error for SGD')
plt.savefig('./figures/t2q2b_1.png')


# find predictions
y_ = []
for p in np.arange(len(X)):
	y_.append(sess.run(y, {x:X[p]}))

    
# plot trained and predicted points
fig = plt.figure(2)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y, 'r.', label='targets')
ax.scatter(X[:,0], X[:,1], y_, 'b.', label='predicted')
ax.set_title('Targets and Predictions')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.legend()
plt.savefig('./figures/t2q2b_2.png')

plt.show()

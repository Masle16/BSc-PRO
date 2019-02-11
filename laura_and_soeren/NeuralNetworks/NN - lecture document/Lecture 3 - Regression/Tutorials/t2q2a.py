#
# Tutorial 2, Question 2: Gradient Descent
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

	
no_iters = 5000
lr = 0.01
SEED = 10
np.random.seed(SEED)

# generate data
no_data = 11*11
X = np.zeros((no_data,2))
Y = np.zeros((no_data, 1))
i = 0
for x1 in np.arange(0, 1.01, 0.1):
    for x2 in np.arange(0, 1.01, 0.1):
        X[i] = [x1, x2]
        Y[i, 0] = 0.5+x1+3*x2**2
        i += 1

# plot data
plt.figure(1)
plt.plot(X[:,0], X[:,1], 'r.')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title('Training Input Data')
plt.axis('equal')
plt.savefig('./figures/t2q2a_1.png')

fig = plt.figure(2)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y[:,0], 'b.')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_title('Targets for Training')
plt.savefig('./figures/t2q2a_2.png')


# Model parameters
w = tf.Variable(np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [None, 2])
d = tf.placeholder(tf.float32, [None, 1])

u = tf.matmul(x,w) + b
y = 4*tf.sigmoid(u) + 0.5
loss = tf.reduce_mean(tf.square(d - y)) # sum of the squares

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
for i in range(no_iters):
  sess.run([w_new, b_new], {x: X, d: Y})
  err.append(sess.run(loss, {x: X, d: Y}))

  if i%100 == 0:
    print('iter: %d, mse: %g'%(i, err[i]))


# plot learning curves
plt.figure(3)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.title('Training Error for GD')
plt.savefig('./figures/t2q2a_3.png')


# evaluate training accuracy
w_, b_, y_, loss_ = sess.run([w, b, y, loss], {x: X, d: Y})
print("w: %s b: %s mse: %g"%(w_, b_, loss_))
    
# plot trained and predicted points
fig = plt.figure(4)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y[:,0], 'r.', label='targets')
ax.scatter(X[:,0], X[:,1], y_[:,0], 'b.', label='predicted')
ax.set_title('Targets and Predictions')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.legend()
plt.savefig('./figures/t2q2a_4.png')

plt.show()

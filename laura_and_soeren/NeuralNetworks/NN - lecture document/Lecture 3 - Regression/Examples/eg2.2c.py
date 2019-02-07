#
# Chapter 2, Example 2: using tf.train
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
Y = np.dot(X, [2.53, -0.47]) + (np.random.rand(6)-0.5)
Y = Y.reshape(6,1)

# Model parameters
w = tf.Variable(np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [None, 2])
d = tf.placeholder(tf.float32, [None, 1])

y = tf.matmul(x,w) + b
loss = tf.reduce_sum(tf.square(d - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) # intialize values 
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err = []
for i in range(no_iters):
  sess.run(train, {x: X, d: Y})
  err.append(sess.run(loss, {x: X, d: Y}))

  if i%10 == 0:
      print('iter: {}, error: {}'.format(i, err[i]))



# evaluate training accuracy
y_, loss_, w_, b_ = sess.run([y, loss, w, b], {x: X, d: Y})
print("y: %s, loss: %s, w: %s b: %s"%(y_, loss_, w_, b_))

plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.savefig('./figures/2.2c_1.png')


pred = sess.run(y, {x: X})

fig = plt.figure(2)
ax = fig.gca(projection = '3d')
plot_original = ax.scatter(X[:,0], X[:,1], Y, 'ro', label='targets')
plot_pred = ax.scatter(X[:,0], X[:,1], pred, 'b^', label='predicted')
X1 = np.arange(-1, 1, 0.1)
X2 = np.arange(-1, 1, 0.1)
X1,X2 = np.meshgrid(X1,X2)
Z = w_[0]*X1 + w_[1]*X2 + b_
regression_plane = ax.plot_surface(X1, X2, Z)
ax.set_zticks([0, 1, 2, 2])
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.title('targets and predictions')
plt.legend()
plt.savefig('./figures/2.2c_2.png')

plt.show()



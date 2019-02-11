#
# Chapter 2, Example 2
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
Y = Y.reshape(6,1)

print(X)
print(Y)

# Model parameters
w = tf.Variable(np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [None, 2])
d = tf.placeholder(tf.float32, [None, 1])

y = tf.matmul(x,w) + b
loss = tf.reduce_mean(tf.square(d - y)) # mean square error

# optimizer
grad_w = -tf.matmul(tf.transpose(x), d - y)
grad_b = -tf.reduce_sum(d - y)
w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) # intialize values

# print initial weights
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err = []
for i in range(no_iters):

    if i < 2:
        y_, loss_ = sess.run([y, loss], {x: X, d: Y})
        print('iter: {}'.format(i+1))
        print('y: {}'.format(y_))
        print('loss: {}'.format(loss_))
                

    sess.run([w_new, b_new], {x: X, d: Y})
    err.append(sess.run(loss, {x: X, d: Y}))

    if i < 2:
        w_, b_ = sess.run([w, b])
        print('w: {}, b: {}'.format(w_, b_))

    if i%10 == 0:
        print('iter: %d, mse: %g'%(i, err[i]))

  


# evaluate final weights and training error
y_, loss_, w_, b_ = sess.run([y, loss, w, b], {x: X, d: Y})
print("w: %s b: %s"%(w_, b_))
print("y: %s, mse: %g"%(y_, loss_))


# plot learning curves
plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.savefig('./figures/2.2a_1.png')


pred = sess.run(y, {x: X})

fig = plt.figure(2)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y, 'ro', label='targets')
ax.scatter(X[:,0], X[:,1], pred, 'b^', label='predicted')

X1 = np.arange(-1, 1, 0.1)
X2 = np.arange(-1, 1, 0.1)
X1,X2 = np.meshgrid(X1,X2)
Z = w_[0]*X1 + w_[1]*X2 + b_
ax.plot_surface(X1, X2, Z)

ax.set_zticks([0, 1, 2, 2])
ax.set_xticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.title('targets and predictions')
plt.legend()
plt.savefig('./figures/2.2a_2.png')

plt.show()



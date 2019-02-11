#
# Tutorial 2, Question 3: perceptron
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

	
no_iters = 1000
lr = 0.05

SEED = 10
np.random.seed(SEED)

# generate data
no_data = 25
X = np.random.rand(no_data,2)
Y = 1.5+3.3*X[:,0]-2.5*X[:,1]+0.2*X[:,0]*X[:,1]
Y = Y.reshape(no_data,1)


# Model parameters
w = tf.Variable(np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [None, 2])
d = tf.placeholder(tf.float32, [None, 1])

u = tf.matmul(x,w) + b
y = 6*tf.sigmoid(u) - 1.0
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
plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('iterations')
plt.ylabel('mean square error')
plt.title('learning curve for perceptron')
plt.savefig('./figures/t2q3b_1.png')


# evaluate training accuracy
w_, b_, y_, loss_ = sess.run([w, b, y, loss], {x: X, d: Y})
print("w: %s b: %s"%(w_, b_))
print("mse: %g"%loss_)
    
# plot trained and predicted points
fig = plt.figure(2)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y[:,0], color='blue', marker='x', label='targets')
ax.scatter(X[:,0], X[:,1], y_[:,0], color='red', marker='.', label='predicted')
ax.set_title('Targets and Predictions')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.legend()
plt.savefig('./figures/t2q3b_2.png')


# plot the learned function
fig = plt.figure(3)
ax = fig.gca(projection = '3d')
X1 = np.arange(0, 1, 0.05)
X2 = np.arange(0, 1, 0.05)
X1,X2 = np.meshgrid(X1,X2)
Z = b_+w_[0]*X1+w_[1]*X2
Z = 6/(1+np.exp(-Z))-1.0
regression_plane = ax.plot_surface(X1, X2, Z)
ax.xaxis.set_major_locator(ticker.IndexLocator(base = 0.2, offset=0.0))
ax.yaxis.set_major_locator(ticker.IndexLocator(base = 0.2, offset=0.0))
ax.set_title('Function Learned by Perceptron')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.savefig('./figures/t2q3b_3.png')



plt.show()


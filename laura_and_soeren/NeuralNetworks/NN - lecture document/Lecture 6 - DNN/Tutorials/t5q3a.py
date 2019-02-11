#
# Tutorial 5, Question 3
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    

# lr = 1e-4
lr = 0.01
num_iters = 10000

num_features = 2
num_hidden1 = 4
num_hidden2 = 3


seed = 100
np.random.seed(seed)
tf.set_random_seed(seed)

# generate data
X = np.zeros((9*9, num_features))
p = 0
for i in np.arange(-1, 1.001, 0.25):
    for j in np.arange(-1, 1.001, 0.25):
        X[p] = [i, j]
        p += 1

np.random.shuffle(X)
Y = np.zeros((9*9, 1))
Y[:,0] = 0.8*X[:,0]**2 - X[:,1]**3 + 2.5*X[:,0]*X[:,1]


plt.figure(1)
plt.plot(X[:,0], X[:,1], 'rx')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Training inputs')
plt.savefig('./figures/t5q3a_1.png')

# plot targets
fig = plt.figure(2)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y[:,0], color='blue', marker='.')
ax.set_title('Targets for training')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
plt.savefig('./figures/t5q3a_2.png')

#Define variables:
W1 = tf.Variable(tf.truncated_normal([num_features, num_hidden1],
                            stddev=1.0 / np.sqrt(num_features) , dtype=tf.float32))
b1 = tf.Variable(tf.zeros([num_hidden1]), dtype=tf.float32)
W2 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2],
                            stddev=1.0 / np.sqrt(num_hidden1) , dtype=tf.float32))
b2 = tf.Variable(tf.zeros([num_hidden2]), dtype=tf.float32)
W3 = tf.Variable(tf.truncated_normal([num_hidden2, 1],
                            stddev=1.0 / np.sqrt(num_hidden2) , dtype=tf.float32))
b3 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [None, X.shape[1]])
d = tf.placeholder(tf.float32, [None, Y.shape[1]])

u1 = tf.matmul(x, W1) + b1
h1 = tf.nn.relu(u1)
u2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.relu(u2)
y = tf.matmul(h2, W3) + b3

cost = tf.reduce_mean(tf.square(d - y))

grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = tf.gradients(cost, [W1, b1, W2, b2, W3, b3])

W3_new, b3_new = W3.assign(W3 - lr*grad_W3), b3.assign(b3 - lr*grad_b3)
W2_new, b2_new = W2.assign(W2 - lr*grad_W2), b2.assign(b2 - lr*grad_b2)
W1_new, b1_new = W1.assign(W1 - lr*grad_W1), b1.assign(b1 - lr*grad_b1)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) 

err = []
for i in range(num_iters):
  sess.run([W1_new, b1_new, W2_new, b2_new, W3_new, b3_new], {x:X, d:Y})
  err.append(sess.run(cost, {x:X, d:Y}))

  if not i%100:
      print('epoch:%d, mse:%g'%(i,err[i]))
                    


# plot learning curves
plt.figure(3)
plt.plot(range(num_iters), err)
plt.xlabel('iterations')
plt.ylabel('error (mse)')
plt.title('GD learning')
plt.savefig('figures/t5q3a_3.png')

# find predictions 
pred = sess.run(y, {x: X})
print('mse: %g'%sess.run(cost, {x: X, d: Y}))

# plot targets and predicted points
fig = plt.figure(4)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y[:,0], color='blue', marker='.', label='targets')
ax.scatter(X[:,0], X[:,1], pred[:,0], color='red', marker='x', label='predictions')
ax.set_title('Targets and Predictions')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.legend()
plt.savefig('./figures/t5q3a_4.png')


plt.show()

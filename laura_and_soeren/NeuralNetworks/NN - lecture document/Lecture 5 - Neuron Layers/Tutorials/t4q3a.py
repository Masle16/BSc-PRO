#
# Tutorial 4, Question 3: GD
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')




lr = 0.05
no_iters = 2000

SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)

X = np.array([[0.50, 0.23],[0.20, 0.76], [0.17, 0.09], [0.69, 0.95],
     [0.00, 0.51], [0.81, 0.61], [0.72, 0.29], [0.92, 0.72]])
Y = np.array([[0.16, 0.74], [0.49, 0.97], [0.01, 0.26], [1.19, 1.70],
     [0.13, 0.52], [0.77, 1.48], [0.40, 1.04], [1.14, 1.70]])

no_features = X.shape[1]
no_labels = Y.shape[1]
no_data = X.shape[0]

# Model parameters
w = tf.Variable(tf.truncated_normal([no_features, no_labels],stddev=1.0 / np.sqrt(no_features)))
b = tf.Variable(tf.zeros([no_labels]))

# Model input and output
x = tf.placeholder(tf.float32, X.shape)
d = tf.placeholder(tf.float32, Y.shape)

u = tf.matmul(x, w) + b
y = 2.0*tf.sigmoid(u)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(d - y), axis=1))

dy = y*(1 - y/2.0)

grad_u = -(d - y)*dy
grad_w = tf.matmul(tf.transpose(x), grad_u)
grad_b = tf.reduce_sum(grad_u, axis = 0)


w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

# print initial weights
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

# training iterations
cost = []
for i in range(no_iters):

  if (i == 0):
    u_, y_, dy_, loss_, grad_u_ = sess.run([ u, y, dy, loss, grad_u], {x:X, d:Y})

    print('iter: {}'.format(i+1))
    print('u: {}'.format(u_))
    print('y: {}'.format(y_))
    print('loss: {}'.format(loss_))
    print('dy: {}'.format(dy_))
    print('grad_u: {}'.format(grad_u_))
    
    
  sess.run([w_new, b_new], {x:X, d:Y})
  loss_ = sess.run(loss, {x:X, d:Y})
  cost.append(loss_)

  if (i == 0):
    w_, b_ = sess.run([w, b])
    print('w: {}, b: {}'.format(w_, b_))

  if not i%100:
    print('epoch: %d, mse: %g'%(i,cost[i]))

# evaluate training accuracy
w_, b_, loss_ = sess.run([w, b, loss], {x:X, d:Y})
print("w: %s b: %s"%(w_, b_))
print("mse: %g"%loss_)

# plot learning curves
plt.figure(1)
plt.plot(range(no_iters), cost)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.title('gd with learning rate %g'%lr)
plt.savefig('./figures/t4q3a_1.png')

pred = sess.run(y, {x:X})

plt.figure(2)
plot_targets = plt.plot(Y[:,0], Y[:,1], 'b^', label='targets')
plot_pred = plt.plot(pred[:,0], pred[:,1], 'ro', label='predicted')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('target and predicted outputs')
plt.legend()
plt.savefig('./figures/t4q3a_2.png')

plt.show()


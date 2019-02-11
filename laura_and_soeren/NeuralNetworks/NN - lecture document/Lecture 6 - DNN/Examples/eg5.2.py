#
# Chapter 5, example 2
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
num_features = 2
num_labels = 2
num_hidden = 3
num_data = 8

lr = 0.05
num_iters = 20000

SEED = 10
np.random.seed(SEED)

# data
# generate training data
X = np.random.rand(num_data, num_features)
Y = 2*np.random.rand(num_data, num_labels) - 1

print('x:{}'.format(X))
print('y:{}'.format(Y))


# initialization routines for bias and weights
def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))

def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                            high=np.sqrt(6. / (n_in + n_out)),
                                            size=(n_in, n_out)))
    if logistic == True:
        W_values *= 4
    return(tf.Variable(W_values, dtype=tf.float32))

#Define variables:
V = init_weights(num_hidden, num_labels)
c = init_bias(num_labels)
W = init_weights(num_features, num_hidden)
b = init_bias(num_hidden)

# Model input and output
x = tf.placeholder(tf.float32, X.shape)
d = tf.placeholder(tf.float32, Y.shape)

z = tf.matmul(x, W) + b
h = tf.nn.sigmoid(z)
y = tf.matmul(h, V) + c


cost = tf.reduce_mean(tf.reduce_sum(tf.square(d - y),axis=1))


grad_u = -(d - y)
grad_V = tf.matmul(tf.transpose(h), grad_u)
grad_c = tf.reduce_sum(grad_u, axis=0)

dh = h*(1-h)
grad_z = tf.matmul(grad_u, tf.transpose(V))*dh
grad_W = tf.matmul(tf.transpose(x), grad_z)
grad_b = tf.reduce_sum(grad_z, axis=0)

W_new = W.assign(W - lr*grad_W)
b_new = b.assign(b - lr*grad_b)
V_new = V.assign(V - lr*grad_V)
c_new = c.assign(c - lr*grad_c)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
W_, b_ = sess.run([W, b])
print('W: {}, b: {}'.format(W_, b_))
V_, c_ = sess.run([V, c])
print('V:{}, c:{}'.format(V_, c_))

err = []
for i in range(num_iters):

  if i == 0:
    z_, h_, y_, cost_, grad_u_, dh_, grad_z_ = sess.run([ z, h, y, cost, grad_u, dh, grad_z], {x:X, d:Y})
    print('iter: {}'.format(i+1))
    print('z: {}'.format(z_))
    print('h: {}'.format(h_))
    print('y: {}'.format(y_))
    print('grad_u: {}'.format(grad_u_))
    print('dh: {}'.format(dh_))
    print('grad_z:{}'.format(grad_z_))
    print('cost: {}'.format(cost_))

  sess.run([W_new, b_new, V_new, c_new], {x:X, d:Y})
  cost_ = sess.run(cost, {x:X, d:Y})
  err.append(cost_)

  if i == 0:
    W_, b_, V_, c_ = sess.run([W, b, V, c])
    print('V: {}, c: {}'.format(V_, c_))
    print('W: {}, b: {}'.format(W_, b_))
                    
  if not i%1000:
    print('epoch:{}, error:{}'.format(i,err[i]))
                    

y_ = sess.run(y, {x: X})
print('y:{}'.format(y_))

# plot learning curves
plt.figure(1)
plt.plot(range(num_iters), err)
plt.xlabel('iterations')
plt.ylabel('mean square error')
plt.title('GD learning')
plt.savefig('figures/5.2_1.png')

# plot trained and predicted points
plt.figure(2)
plot_targets = plt.plot(Y[:,0], Y[:,1], 'b^', label='targeted')
plot_pred = plt.plot(y_[:,0], y_[:,1], 'ro', label='predicted')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('targets and predicted outputs')
plt.legend()
plt.savefig('./figures/5.2_2.png')

plt.show()

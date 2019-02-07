#
# Tutorial 5, Question 1: gradient descent
#

import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    

lr = 0.05
num_iters = 5000

# data
# generate training data
X = np.array([[1.0, 3.0], [-2.0, -2.0]])
Y = np.array([[0.0, 1.0], [1.0, 0.0]])

print('x: %s, y: %s'%(X, Y))


#Define variables:
V = tf.Variable(np.array([[1.0, 1.0], [0, -2]]) , dtype=tf.float32)
c = tf.Variable(np.array([-2.0, 3.0]), dtype=tf.float32)
W = tf.Variable(np.array([[1.0, 2.0],[-2.0, 0]]), dtype=tf.float32)
b = tf.Variable(np.array([3.0, -1.0]), dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, X.shape)
d = tf.placeholder(tf.float32, Y.shape)

z = tf.matmul(x, W) + b
h = tf.nn.sigmoid(z)
u = tf.matmul(h, V) + c
y = tf.nn.sigmoid(u)


cost = tf.reduce_mean(tf.reduce_sum(tf.square(d - y),axis=1))

dy = y*(1-y)
grad_u = -(d - y)*dy
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

err = []
for i in range(num_iters):

  if i == 0:
    z_, h_, u_, y_, cost_, dy_, grad_u_, dh_, grad_z_ = sess.run([ z, h, u, y, cost, dy, grad_u, dh, grad_z], {x:X, d:Y})
    print('iter: {}'.format(i+1))
    print('z: {}'.format(z_))
    print('h: {}'.format(h_))
    print('u: {}'.format(u_))
    print('y: {}'.format(y_))
    print('dy: {}'.format(dy_))
    print('grad_u: {}'.format(grad_u_))
    print('dh: {}'.format(dh_))
    print('grad_z:{}'.format(grad_z_))
    print('cost: {}'.format(cost_))

  sess.run([W_new, b_new, V_new, c_new], {x:X, d:Y})
  cost_ = sess.run(cost, {x:X, d:Y})
  err.append(cost_)

  if i == 0:
    V_, c_, W_, b_, = sess.run([V, c, W, b])
    print('V: {}, c: {}'.format(V_, c_))
    print('W: {}, b: {}'.format(W_, b_))
                    
  if not i%100:
    print('epoch:{}, error:{}'.format(i,err[i]))
                    
W_, b_, V_, c_= sess.run([W, b, V, c])
print('V: {}, c: {}'.format(V_, c_))
print('W: {}, b: {}'.format(W_, b_))

y_, cost_ = sess.run([y, cost], {x: X, d: Y})
print('y:{}'.format(y_))
print('mse: %g'%cost_)

# plot learning curves
plt.figure(1)
plt.plot(range(num_iters), err)
plt.xlabel('epochs')
plt.ylabel('mean square error')
plt.title('GD learning')
plt.savefig('figures/t5q1a_1.png')


plt.show()

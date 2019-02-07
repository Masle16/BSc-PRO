#
# Tutorial 5, Question 2
#

import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    

lr = 0.1
num_iters = 500

num_features = 2
num_classes = 3
num_hidden = 3

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# data
# generate training data
X = np.array([[1.0, 1.0], [0.0, 1.0], [3.0, 4.0], [2.0, 2.0], [2.0, -2.0], [-2.0, -3.0]])
Y = np.array([0, 0, 1, 1, 2, 2])
K = np.array([[1, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 1]]).astype(float)

print('x: %s, y: %s'%(X, Y))


#Define variables:
V = tf.Variable(tf.truncated_normal([num_hidden, num_classes],
                            stddev=1.0 / np.sqrt(float(num_hidden)) , dtype=tf.float32))
c = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)
W = tf.Variable(tf.truncated_normal([num_features, num_hidden],
                            stddev=1.0 / np.sqrt(float(num_features)) , dtype=tf.float32))
b = tf.Variable(tf.zeros([num_hidden]), dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, X.shape)
k = tf.placeholder(tf.float32, K.shape)

z = tf.matmul(x, W) + b
h = tf.nn.sigmoid(z)
u = tf.matmul(h, V) + c
p = tf.exp(u)/tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)
y = tf.argmax(p, axis=1)


loss = -tf.reduce_sum(tf.log(p)*k)
err = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(k, 1), y), tf.int32))

grad_u = -(k - p)
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

# print initial weights
V_, c_, W_, b_ = sess.run([V, c, W, b])
print('W: {}, b: {}'.format(W_, b_))
print('V: {}, c: {}'.format(V_, c_))

loss_, err_ = [], []
for i in range(num_iters):
  if i == 0:
    
    z_, h_, u_, p_, y_, l_, grad_u_, dh_, grad_z_ = sess.run([ z, h, u, p, y, loss, grad_u, dh, grad_z], {x:X, k:K})
    print('iter: {}'.format(i+1))
    print('z: {}'.format(z_))
    print('h: {}'.format(h_))
    print('u: {}'.format(u_))
    print('f(u): {}'.format(p_))
    print('y: {}'.format(y_))
    print('entropy: {}'.format(l_))
    print('grad_u: {}'.format(grad_u_))
    print('dh: {}'.format(dh_))
    print('grad_z:{}'.format(grad_z_))

  sess.run([W_new, b_new, V_new, c_new], {x:X, k:K})
  err_.append(sess.run(err, {x:X, k:K}))
  loss_.append(sess.run(loss, {x:X, k:K}))

  if i == 0:
    V_, c_, W_, b_ = sess.run([V, c, W, b])
    print('V: {}, c: {}'.format(V_, c_))
    print('W: {}, b: {}'.format(W_, b_))
                    
  if not i%100:
    print('epoch:%d, error:%g, entropy:%g'%(i,err_[i], loss_[i]))

# print final weights                   
V_, c_, W_, b_ = sess.run([V, c, W, b])
print('V: {}, c: {}'.format(V_, c_))
print('W: {}, b: {}'.format(W_, b_))

print('entropy: %g'%loss_[num_iters-1])
print('error: %g'%err_[num_iters-1])

# find predictions
y_ = sess.run(y, {x: X})
print('y:{}'.format(y_))

# plot learning curves
plt.figure(1)
plt.plot(range(num_iters), err_)
plt.xlabel('iterations')
plt.ylabel('classification error')
plt.title('GD learning')
plt.savefig('figures/t5q2_1.png')

plt.figure(2)
plt.plot(range(num_iters), loss_)
plt.xlabel('iterations')
plt.ylabel('entropy loss')
plt.title('GD learning')
plt.savefig('figures/t5q2_2.png')

plt.show()

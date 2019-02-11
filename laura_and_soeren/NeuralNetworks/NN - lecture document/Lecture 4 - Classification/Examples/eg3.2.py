#
# Chapter 3, example 2
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
lr = 0.4

SEED = 10
np.random.seed(SEED)


# training data
x_train = np.array([[1.33, 0.72], [-1.55, -0.01], [0.62, -0.72],
    [0.27, 0.11], [0.0, -0.17], [0.43, 1.2], [-0.97, 1.03], [0.23, 0.45]])
y_train = np.array([0, 1, 1, 1, 1, 0, 0, 0]).reshape(8,1)

print(x_train)
print(y_train)
print(lr)

# Model parameters
w = tf.Variable(np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, x_train.shape)
d = tf.placeholder(tf.int32, y_train.shape)


u = tf.matmul(x, w) + b
f_u = tf.sigmoid(u)
d_float = tf.cast(d, tf.float32)

loss = -tf.reduce_sum(d_float*tf.log(f_u) + (1-d_float)*tf.log(1-f_u))
class_err = tf.reduce_sum(tf.cast(tf.not_equal(f_u > 0.5, y_train), tf.int32))

grad_u = -(d_float - f_u)
grad_w = tf.matmul(tf.transpose(x), grad_u)
grad_b = tf.reduce_sum(grad_u)

w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err = []
c_err = []
for i in range(no_iters):
    u_, f_u_, loss_, c_err_, w_, b_ = sess.run([u, f_u, loss, class_err, w_new, b_new], {x: x_train, d: y_train})

    if (i == 0):
        print('u:{}'.format(u_))
        print('f_u:{}'.format(f_u_))
        print('y:{}'.format(f_u_ > 0.5))
        print('loss:{}'.format(loss_))
        print('error:{}'.format(c_err_))
        print('w: {}, b: {}'.format(w_, b_))

    err.append(loss_)
    c_err.append(c_err_)

    if (i%10 == 0):
        print('iter: {}, cost: {}'.format(i, err[i]))


# evaluate training accuracy
print('w: {}, b: {}'.format(w_, b_))

print(f_u_ > 0.5)

plt.figure(1)
plt.plot(x_train[y_train[:,0]==1,0], x_train[y_train[:,0]==1,1],'bx', label ='class A')
plt.plot(x_train[y_train[:,0]==0,0],x_train[y_train[:,0]==0,1],'ro', label='class B')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('training data')
plt.legend()
plt.savefig('./figures/3.2_1.png')

plt.figure(2)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/3.2_2.png')


plt.figure(3)
plt.plot(range(no_iters), c_err)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/3.2_3.png')

x1 = np.arange(-2, 2, 0.1)
x2 = -(x1*w_[0] + b_)/w_[1]

plt.figure(4)
plt.plot(x_train[y_train[:,0]==1,0], x_train[y_train[:,0]==1,1],'bx', label ='class A')
plt.plot(x_train[y_train[:,0]==0,0],x_train[y_train[:,0]==0,1],'ro', label='class B')
plt.plot(x1, x2, '-')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('decision boundary')
plt.legend()
plt.savefig('./figures/3.2_4.png')

plt.show()



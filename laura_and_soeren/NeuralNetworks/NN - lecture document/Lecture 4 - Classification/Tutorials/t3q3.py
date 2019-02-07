#
# Tutorial 3, Question 3
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


no_iters = 250
lr = 0.4

SEED = 10
np.random.seed(SEED)

# training data
x_train = np.array([[-1.75, 0.34, 1.15],
     [-0.25, 0.98, 0.51],
     [0.22, -1.07, -0.19],
     [0.26, -0.46, 0.44],
     [-0.58, 0.82, 0.67],
     [-0.1, -0.53, 1.03],
     [-0.44, -1.12, 1.62],
     [1.54, -0.25, -0.84],
     [0.18, 0.94, 0.73],
     [1.36, -0.33, 0.06]])
y_train = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 0]).reshape(10,1)

print(x_train)
print(y_train)
print(lr)

# Model parameters
w = tf.Variable(np.random.rand(3,1), dtype=tf.float32)
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
        print('y: %d'.format((f_u_ > 0.5).astype(int)))
        print('loss:{}'.format(loss_))
        print('error:{}'.format(c_err_))
        print('w: {}, b: {}'.format(w_, b_))

    err.append(loss_)
    c_err.append(c_err_)

    if (i%10 == 0):
        print('iter: %d, cost: %g, error: %d'%(i, err[i], c_err[i]))


# evaluate training accuracy
print('w: {}, b: {}'.format(w_, b_))

print('f(u): %s'%f_u_)
y_ = f_u_ > 0.5
print(y_.astype(int))

# plot input data
fig = plt.figure(1)
ax = fig.gca(projection = '3d')
ax.scatter(x_train[y_train[:,0]==1,0], x_train[y_train[:,0]==1,1], x_train[y_train[:,0]==1,2],'b^', label='class 1')
ax.scatter(x_train[y_train[:,0]==0,0], x_train[y_train[:,0]==0,1], x_train[y_train[:,0]==0,2], 'ro', label='class 2')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('data points')
ax.legend()
plt.savefig('./figures/t3q3_1.png')

# plot learning curves
plt.figure(2)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/t3q3_2.png')


plt.figure(3)
plt.plot(range(no_iters), c_err)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/t3q3_3.png')


# plot decision boundary
fig = plt.figure(4)
ax = fig.gca(projection='3d')
ax.scatter(x_train[y_train[:,0]==1,0], x_train[y_train[:,0]==1,1], x_train[y_train[:,0]==1,2],'b^', label='class 1')
ax.scatter(x_train[y_train[:,0]==0,0], x_train[y_train[:,0]==0,1], x_train[y_train[:,0]==0,2], 'ro', label='class 2')
x1, x2 = np.meshgrid(np.arange(-2, 2, 0.1),np.arange(-2, 2, 0.1))
x3 = -(w_[0]*x1 + w_[1]*x2 + b_)/w_[2]
decision_boundary = ax.plot_surface(x1, x2, x3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('Decision boundary')
ax.legend()
plt.savefig('./figures/t3q3_4.png')


plt.show()



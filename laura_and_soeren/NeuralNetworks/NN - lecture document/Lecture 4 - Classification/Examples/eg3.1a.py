#
# Chapter 3, example 1
#

import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
	
no_iters = 30
lr = 0.4

SEED = 10
np.random.seed(SEED)


# training data
x_train = np.array([[1.0, 2.5], [2.0, -1.0], [1.5, 3.0],
	[0.0, -1.5], [-3.5, 1.0], [2.5, 0.0], [0.5, 1.5], [0.0, -2.0]])
y_train = np.array([1, 0, 1, 0, 1, 0, 0, 0])

print(x_train)
print(y_train)
print(lr)

# Model parameters
w = tf.Variable(np.random.rand(2), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
d = tf.placeholder(tf.int32)

u = tf.tensordot(x,w, axes=1) + b
y = tf.where(tf.greater(u, 0), 1, 0)

delta = d - y
delta = tf.cast(delta, tf.float32)

w_new = w.assign(w + lr*delta*x)
b_new = b.assign(b + lr*delta)


# initialize the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# print initial weights
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

# training loop
err = []
idx = np.arange(len(x_train))
for i in range(no_iters):
    np.random.shuffle(idx)
    x_train, y_train = x_train[idx], y_train[idx]
    
    err_ = 0
    for p in np.arange(len(x_train)):
        u_, y_, w_, b_ = sess.run([u, y, w_new, b_new], {x: x_train[p], d: y_train[p]})
        err_ += y_ != y_train[p]

        if (i == 0):
            print('p: {}'.format(p+1))
            print('x: {}'.format(x_train[p]))
            print('d: {}'.format(y_train[p]))
            print('u: {}'.format(u_))
            print('y: {}'.format(y_))
            print('w: {}, b: {}'.format(w_, b_))

    err.append(err_)
       
    print('iter: {}, error: {}'.format(i+1, err[i]))
        
 
	
# print final weights
print('w: {}, b: {}'.format(w_, b_))

# plot the learning curves
plt.figure(2)
plt.plot(range(no_iters), err)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.yticks([0, 1, 2])
plt.savefig('./figures/3.1a_2.png')

# find predicctions
pred = []
for p in np.arange(len(x_train)):
	pred.append(sess.run(y, {x: x_train[p]}))
print(y_train, pred)

# plot the training data
plt.figure(1)
plt.plot(x_train[y_train==1,0], x_train[y_train==1,1],'bx', label ='class A')
plt.plot(x_train[y_train==0,0],x_train[y_train==0,1],'ro', label='class B')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('training data')
plt.legend()
plt.savefig('./figures/3.1a_1.png')

# plot the decision boundary
x1 = np.arange(-4, 4, 0.1)
x2 = -(x1*w_[0] + b_)/w_[1]

plt.figure(3)
plt.plot(x_train[y_train==1,0], x_train[y_train==1,1],'bx', label ='class A')
plt.plot(x_train[y_train==0,0],x_train[y_train==0,1],'ro', label='class B')
plt.plot(x1, x2, '-')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('decision boundary')
plt.legend()
plt.savefig('./figures/3.1a_3.png')

plt.show()



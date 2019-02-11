#
# Tutorial 5, Question 1b: stochastic gradient descent
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

seed = 10
np.random.seed(seed)

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
x = tf.placeholder(tf.float32)
d = tf.placeholder(tf.float32)

z = tf.tensordot(tf.transpose(W), x, axes=1) + b
h = tf.nn.sigmoid(z)
u = tf.tensordot(tf.transpose(V), h, axes=1) + c
y = tf.nn.sigmoid(u)


cost = tf.reduce_sum(tf.square(d - y))

dy = y*(1-y)
grad_u = -(d - y)*dy

grad_V = tf.tensordot(h, grad_u, axes=0)
grad_c = grad_u

dh = h*(1-h)
grad_z = tf.tensordot(V, grad_u, axes=1)*dh

grad_W = tf.tensordot(x, grad_z, axes=0)
grad_b = grad_z

W_new = W.assign(W - lr*grad_W)
b_new = b.assign(b - lr*grad_b)
V_new = V.assign(V - lr*grad_V)
c_new = c.assign(c - lr*grad_c)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

err = []
idx = np.arange(2)
for i in range(num_iters):
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    
    cost_ = []
    for p in [0, 1]:

        if i == 0:
            z_, h_, u_, y_, cos_, dy_, grad_u_, dh_, grad_z_ = sess.run([ z, h, u, y, cost, dy, grad_u, dh, grad_z], {x:X[p], d:Y[p]})
            print('iter: %d, pattern: %d'%(i+1, p+1))
            print('x: {}'.format(X[p]))
            print('y: {}'.format(Y[p]))
            print('z: {}'.format(z_))
            print('h: {}'.format(h_))
            print('u: {}'.format(u_))
            print('y: {}'.format(y_))
            print('cost: {}'.format(cos_))
            print('dy: {}'.format(dy_))
            print('grad_u: {}'.format(grad_u_))
            print('dh: {}'.format(dh_))
            print('grad_z:{}'.format(grad_z_))
            
        sess.run([W_new, b_new, V_new, c_new], {x:X[p], d:Y[p]})
        cost_.append(sess.run(cost, {x:X[p], d:Y[p]}))

        if i == 0:
            V_, c_, W_, b_ = sess.run([V, c, W, b])
            print('V: {}, c: {}'.format(V_, c_))
            print('W: {}, b: {}'.format(W_, b_))

    err.append(np.mean(cost_))

    if not i%100:
        print('epoch:%d, mse: %g'%(i,err[i]))
                    
V_, c_, W_, b_ = sess.run([V, c, W, b])
print('V: {}, c: {}'.format(V_, c_))
print('W: {}, b: {}'.format(W_, b_))
for p in [0, 1]:
    y_ = sess.run(y, {x: X[p]})
    print('x: %s, y: %s'%(X[p], y_))
print('mse: %g'%err[num_iters - 1])

# plot learning curves
plt.figure(1)
plt.plot(range(num_iters), err)
plt.xlabel('iterations')
plt.ylabel('mean square error')
plt.title('SGD learning')
plt.savefig('figures/t5q1b_1.png')


plt.show()

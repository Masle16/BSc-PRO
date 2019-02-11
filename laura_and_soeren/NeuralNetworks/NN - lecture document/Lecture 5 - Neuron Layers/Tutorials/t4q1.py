#
# Tutorial 4, Question 1
#


import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

num_iters = 1500
num_features = 2
num_classes = 3
num_data = 18

lr = 0.05

SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)

# input data
X = np.array([[0, 4],[-1, 3],[2, 3], [-2, 2],[0, 2], [1, 2],
              [-1, 2],[-3, 1],[-1, 1],[2, 1],[4, 1],[-2, 0],
             [1, 0],[3, 0],[-3, -1],[-2, -1],[2, -1],[4, -1]])
            
Y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2]).astype(int)
K = np.zeros((num_data, num_classes)).astype(float)
for p in range(len(Y)):
    K[p,Y[p]] = 1
    

print(X)
print(Y)
print(lr)

plt.figure(1)
plot_pred = plt.plot(X[Y==0,0], X[Y==0,1], 'b^', label='class 1')
plot_original = plt.plot(X[Y==1,0], X[Y==1,1], 'ro', label='class 2')
plot_original = plt.plot(X[Y==2,0], X[Y==2,1], 'gx', label='class 3')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('data points')
plt.legend()
plt.savefig('./figures/t4q1_1.png')


# Model parameters
w = tf.Variable(tf.truncated_normal([2, 3],stddev=1.0 / np.sqrt(4)))
b = tf.Variable(tf.zeros([num_classes]))

# Model input and output
x = tf.placeholder(tf.float32, X.shape)
k = tf.placeholder(tf.float32, K.shape)

u = tf.matmul(x, w) + b
p = tf.exp(u)/tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)

y = tf.argmax(p, axis=1)

loss = -tf.reduce_sum(tf.log(p)*k)
err = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(k, 1), y), tf.int32))

grad_u = -(k - p)
grad_w = tf.matmul(tf.transpose(x), grad_u)
grad_b = tf.reduce_sum(grad_u, axis = 0)

w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

loss_, err_ = [], []
for i in range(num_iters):

  if (i == 0):
    u_, p_, y_, l_, e_, du_ = sess.run([u, p, y, loss, err, grad_u], {x: X, k:K})
    print('iter: {}'.format(i+1))
    print('u: {}'.format(u_))
    print('p: {}'.format(p_))
    print('y: {}'.format(y_))
    print('grad_u: {}'.format(du_))
    print('loss: {}'.format(l_))
    print('error: {}'.format(e_))
 
  sess.run([w_new, b_new], {x:X, k:K})
  l, e = sess.run([loss, err], {x:X, k:K})
  loss_.append(l)
  err_.append(e)

  if (i == 0):
    w_, b_ = sess.run([w, b], {x: X, k:K})
    print('w: {}, b: {}'.format(w_, b_))

  if not i%100:
    print('epoch: %d, loss: %g, error: %d'%(i,loss_[i], err_[i]))

# evaluate training accuracy
w_, b_, p_, y_, l_, e_ = sess.run([w, b, p, y, loss, err], {x:X, k:K})
print("w: %s, b: %s"%(w_, b_))
print("p: %s"%p_)
print("y: %s"%y_)
print("loss: %g, error: %g"%(l_, e_))

plt.figure(2)
plt.plot(range(num_iters), loss_)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/t4q1_2.png')

plt.figure(3)
plt.plot(range(num_iters), err_)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/t4q1_3.png')


plt.show()


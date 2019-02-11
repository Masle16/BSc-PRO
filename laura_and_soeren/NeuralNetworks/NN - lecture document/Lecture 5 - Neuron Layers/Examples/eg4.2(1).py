import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


no_features = 3
no_labels = 2
no_data = 8

lr = 0.1
no_iters = 10000

SEED = 10
np.random.seed(SEED)

# generate training data
X = np.random.rand(no_data, no_features)
#Y = np.random.rand(num_data, num_labels)
Y = np.zeros((no_data, no_labels))
Y[:,0] = (X[:,0] + X[:,1]**2 + X[:, 2]**3 + np.random.rand(no_data))/4
Y[:,1] = (X[:,0] + X[:,1] + X[:, 2] + X[:,0]*X[:,1]*X[:,2] + np.random.rand(no_data))/5

print('X = {}'.format(X))
print('Y = {}'.format(Y))
print('alpha = {}'.format(lr))

# Model parameters
w = tf.Variable(np.random.rand(no_features, no_labels)*0.05, dtype=tf.float32)
b = tf.Variable(tf.zeros([no_labels]))

# Model input and output
x = tf.placeholder(tf.float32, X.shape)
d = tf.placeholder(tf.float32, Y.shape)

u = tf.matmul(x, w) + b
y = tf.sigmoid(u)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(d - y), axis=1))

dy = y*(1 - y)

grad_u = -(d - y)*dy
grad_w = tf.matmul(tf.transpose(x), grad_u)
grad_b = tf.reduce_sum(grad_u, axis = 0)
# grad_w, grad_b = tf.gradients(loss, [w, b])

w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

cost = []
for i in range(no_iters):
  if (i < 2 or i == no_iters - 1):
    print('iter: {}'.format(i+1))
    u_, y_, dy_, grad_u_, loss_, w_, b_ = sess.run([ u, y, dy, grad_u, loss, w, b], {x:X, d:Y})
    print('u: {}'.format(u_))
    print('y: {}'.format(y_))
    print('dy: {}'.format(dy_))
    print('grad_u: {}'.format(grad_u_))
    print('m.s.e: {}'.format(loss_))

  sess.run([w_new, b_new], {x:X, d:Y})
  loss_ = sess.run(loss, {x:X, d:Y})
  cost.append(loss_)

  if (i < 2 or i == no_iters - 1):
    print('w: {}, b: {}'.format(w_, b_))

  if not i%200:
    print('epoch:{}, loss:{}'.format(i,cost[i]))

# evaluate training accuracy
curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x:X, d:Y})
print("w: %s b: %s"%(curr_w, curr_b))
print("mse: %g"%curr_loss)

# plot learning curves
plt.figure(1)
plt.plot(range(no_iters), cost)
plt.xlabel('iterations')
plt.ylabel('mean square error')
plt.title('gd with alpha = {}'.format(lr))
plt.savefig('./figures/4.2_1.png')

pred = sess.run(y, {x:X})

plt.figure(2)
plot_targets = plt.plot(Y[:,0], Y[:,1], 'b^', label='targets')
plot_pred = plt.plot(pred[:,0], pred[:,1], 'ro', label='predicted')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('gd outputs')
plt.legend()
plt.savefig('./figures/4.2_2.png')

plt.show()


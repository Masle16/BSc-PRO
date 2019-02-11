#
# Tutorial 3, Question 2b
#



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
lr = 0.1
no_iters = 300
seed = 100
np.random.seed(seed)

#Define inputs and weights
inputs = np.array([[0.8, 0.5, 0.0],
                    [0.9, 0.7, 0.3],
                    [1.0, 0.8, 0.5],
                    [0.0, 0.2, 0.3],
                    [0.2, 0.3, 0.5],
                    [0.4, 0.7, 0.8]])
outputs = np.array([0,0,0,1,1,1])

print(inputs)
print(outputs)
print(lr)

# Model parameters
w = tf.Variable(np.random.rand(3), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
d = tf.placeholder(tf.int32)

u = tf.tensordot(x,w, axes=1) + b
f_u = tf.sigmoid(u)
d_float = tf.cast(d, tf.float32)

# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_float, logits=f_u)
loss = -d_float*tf.log(f_u) - (1 - d_float)*tf.log(1-f_u)
error = tf.cast(tf.not_equal(tf.cast(f_u > 0.5, tf.int32), d), tf.int32)

# grad_w, grad_b = tf.gradients(loss, [w, b])
grad_w = -(d_float - f_u)*x
grad_b = -(d_float - f_u)
w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err, cost = [], []
idx = np.arange(len(inputs))
for i in range(no_iters):
    np.random.shuffle(idx)
    x_train, y_train = inputs[idx], outputs[idx]
    
    err_, cost_ = 0, 0
    for p in np.arange(len(inputs)):

        if i == 0:
            u_, f_u_, loss_, error_ = sess.run([u, f_u, loss, error], {x: x_train[p], d: y_train[p]})
            print('p: %d'%(p+1))
            print('x: %s'%x_train[p])
            print('d: %d'%y_train[p])
            print('u: %g'%u_)
            print('f(u): %g'%f_u_)
            print('loss: %g'%loss_)
            
        sess.run([w_new, b_new], {x: x_train[p], d: y_train[p]})
        err_ += sess.run(error,  {x: x_train[p], d: y_train[p]})
        cost_ += sess.run(loss,  {x: x_train[p], d: y_train[p]})

        if (i == 0):
            w_, b_ = sess.run([w, b])
            print('w: %s, b: %s'%(w_, b_))

    err.append(err_/len(inputs))
    cost.append(cost_/len(inputs))
    if i%10 == 0:
        print('epoch: %d, cost: %g, error: %g'%(i, cost[i], err[i]))
        
 


# plot learning curves
plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('iterations')
plt.ylabel('classification error')
plt.title('SGD learning of a logistic neuron')
plt.savefig('./figures/t3q2b_1.png')

plt.figure(2)
plt.plot(range(no_iters), cost)
plt.xlabel('iterations')
plt.ylabel('entropy')
plt.title('SGD learning of a logistic neuron')
plt.savefig('./figures/t3q2b_2.png')

w_, b_ = sess.run([w, b])
print('w: %s'%w_)
print('b: %g'%b_)

# plot data points and decision boundary
fig = plt.figure(3)
ax = fig.gca(projection='3d')
c1 = ax.scatter(inputs[outputs==0,0],inputs[outputs==0,1],inputs[outputs==0,2],marker='x')
c2 = ax.scatter(inputs[outputs==1,0],inputs[outputs==1,1],inputs[outputs==1,2],marker='x')
X = np.arange(0, 1, 0.1)
Y = np.arange(0, 1, 0.1)
X, Y = np.meshgrid(X,Y)
Z = -(w_[0]*X + w_[1]*Y + b_)/w_[2]
decision_boundary = ax.plot_surface(X, Y, Z)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('Decision boundary in input space')
plt.savefig('./figures/t3q2b_3.png')

plt.show()
    

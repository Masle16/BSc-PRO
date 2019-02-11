#
# Tutorial 3, Question 2a
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
no_iters = 20

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
y = tf.where(tf.greater(u, 0), 1, 0)
delta = d - y
delta = tf.cast(delta, tf.float32)

w_new = w.assign(w + lr*delta*x)
b_new = b.assign(b + lr*delta)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))

err = []
idx = np.arange(len(inputs))
for i in range(no_iters):
    np.random.shuffle(idx)
    x_train, y_train = inputs[idx], outputs[idx]
    
    err_ = 0
    for p in np.arange(len(inputs)):
        u_, y_, w_, b_ = sess.run([u, y, w_new, b_new], {x: x_train[p], d: y_train[p]})
        err_ += y_ != y_train[p]

        if (i == 0):
            print('p: %d'%(p+1))
            print('x: %s'%x_train[p])
            print('d: %d'%y_train[p])
            print('u: %g'%u_)
            print('y: %d'%y_)
            print('w: %s, b: %s'%(w_, b_))

    err.append(err_/len(inputs))
    print('epoch: %d, error: %g'%(i, err[i]))
        
 
print('w: %s'%w_)
print('b: %g'%b_)
print('error: %g'%err[no_iters-1])

# plot learning curves
plt.figure(1)
plt.plot(range(no_iters), err)
plt.xlabel('iterations')
plt.ylabel('classification error')
plt.xticks(np.arange(no_iters//4)*4, np.arange(no_iters//4)*4)
plt.title('Discrete Perceptron Learning')
plt.savefig('./figures/t3q2a_1.png')


# plot data points and decision boundary
fig = plt.figure(2)
ax = fig.gca(projection='3d')
c1 = ax.scatter(inputs[outputs==0,0],inputs[outputs==0,1],inputs[outputs==0,2],marker='x', label='class A')
c2 = ax.scatter(inputs[outputs==1,0],inputs[outputs==1,1],inputs[outputs==1,2],marker='x', label='class B')
X = np.arange(0, 1, 0.1)
Y = np.arange(0, 1, 0.1)
X, Y = np.meshgrid(X,Y)
Z = -(w_[0]*X + w_[1]*Y + b_)/w_[2]
decision_boundary = ax.plot_surface(X, Y, Z)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('Decision boundary in Input Space')
ax.legend()
plt.savefig('./figures/t3q2a_2.png')

plt.show()
    

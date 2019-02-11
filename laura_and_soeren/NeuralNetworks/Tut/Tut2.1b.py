## Learn with GD linear neuron

import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## Learning parameters
no_iters = 200
alpha = tf.Variable(0.01, tf.float32)

SEED = 10
np.random.seed(SEED)

## Training data
X = np.array([[0.09, -0.44, -0.15],
              [0.69, -0.99, -0.76],
              [0.34, 0.65, -0.73],
              [0.15, 0.78, -0.58],
              [-0.63, -0.78, -0.56],
              [0.96, 0.62, -0.66],
              [0.63, -0.45, -0.14],
              [0.88, 0.64, -0.33]])
Y = np.array([-2.57, -2.97, 0.96, 1.04, -3.21, 1.05, -2.39, 0.66])
Y=Y.reshape(8,1)

print('x: %s'%X)
print('d: %s\n'%Y)

## Build the graph
    # 3 input nodes, 1 neuron
w = tf.Variable(np.random.rand(3,1), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

x = tf.placeholder(tf.float32, [None, 3])
d = tf.placeholder(tf.float32, [None, 1])
y = tf.matmul(x,w) + b     # y= Xw+b1_p

loss = tf.reduce_mean(tf.square(d-y))

## Optimize
grad_w = - tf.matmul(tf.transpose(x), (d-y) )
grad_b = - tf.reduce_sum(d-y)

w_new = w.assign(w - alpha*grad_w)
b_new = b.assign(b - alpha*grad_b)



## Start session and Print initial values
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run (init)

w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_,b_))

## Constainers for error
mse = []

## Loops
for i in range(no_iters):

    y_, w_, b_ = sess.run([y,w_new,b_new], {x:X, d:Y})
    mse.append(sess.run([loss], {x:X, d:Y}))
    if (i==0):
        print('First iteration')
        print('i: %d'%(i+1))
        print('y: {}'.format(y_))
        print('w: {}, b: {}'.format(w_, b_))
        print('MSE: {}'.format(mse[i]) )
        print('---------------------')

    if (i%10 == 0):
        print('i: %d'%(i+1))
        print('w: {}, b: {}'.format(w_, b_))
        print('MSE: {}'.format(mse[i]) )
        print('---------------------')

    

## Plot rep vs MSE
plt.figure(1)
plt.plot(range(no_iters), mse)
plt.show()

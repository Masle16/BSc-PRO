## Learn with stocastis linear neuron

import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## Learning parameters
no_iterations = 200
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

## Build the graph
    # w = [x,x,x]^T    
w = tf.Variable(np.random.rand(3), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)

    # Model input and output
x = tf.placeholder(tf.float32, [3]) # x^T = [x,x,x]
d = tf.placeholder(tf.float32)
    
y = tf.tensordot(x,w, axes=1)+b     # u = x^T*W + b
loss = tf.square(d-y)

    # Optimize
w_grad = -(d - y)*x                 # w_grad = -(d-y)*X
b_grad = -(d - y)

w_new = w.assign(w - alpha*w_grad)
b_new = b.assign(b - alpha*b_grad)

    # Start the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

## Start the secssion Get the starting values of w and b
w_,b_ = sess.run([w,b])
print('w: {}, b: {}'.format(w_,b_))

## Container for plotting the mean square error
MSE = []

## Randomize
idx = np.arange(len(X))

## Loop - per_iteration (epoke)
for i in range(no_iterations):
    err_ = []
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    
        # Loop - dataset
    for p in np.arange(len(X)):
        y_, w_, b_, loss_ = sess.run( [y, w_new, b_new, loss], {x: X[p], d: Y[p]}  )
        err_.append(loss_)

        if (i == 0):
            print('iter: %d'%(i+1))
            print('p: %d'%(p+1))
            print('x: {}; d: {}'.format(X[p], Y[p]) )
            print('y: %g'%y_)
            print('w: {}, b: {}'.format(w_,b_))
            print('loss: %g'%(loss_))

    MSE.append( np.mean(err_))

    if (i%10 == 0):
        print('itr: %d, error: %g' %(i,MSE[i]) )
        
print('w: {}, b: {}'.format(w_, b_))
print('mse: %g'%MSE[no_iterations-1])    

## Plot the data

plt.figure(1)
plt.plot(range(no_iterations), MSE)
plt.xlabel('no iterations')
plt.ylabel('Mean Square Error')
plt.show()

## plot the figure, itaration vs mse
## x and y labels, show the figure

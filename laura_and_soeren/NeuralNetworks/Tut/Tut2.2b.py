## Design a SGD perceptron to approximate the function y = 0.5 + x_1 + 4x_2^2 
import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## Generate training data
iters = 500  # Epokes
lr = 0.01    # learning rate

SEED = 10
np.random.seed(SEED)

    # Data is the grid from 0 < x < 1 med 0.1 interval = 11*11= 121 data poits
no_data = 11*11
X = np.zeros((no_data,2))
Y = np.zeros(no_data)
i = 0

for x1 in np.arange(0, 1.01, 0.1):
    for x2 in np.arange(0, 1.01, 0.1):
        X[i] = [x1,x2]
        Y[i] = 0.5 + x1 + 3*x2**2
        i+=1

amp = Y.max()-Y.min()   # Data amplitude
offset = Y.min()        # Data ofset from 0

    ## plot training data
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:,0], X[:,1], Y)
#plt.show()

## Build the graph
    # Weights, bias, i/o, target
w = tf.Variable(np.random.rand(2,1), dtype = tf.float32)
b = tf.Variable([0.], dtype = tf.float32)

x = tf.placeholder(tf.float32, [2])    # [?,2] dimension
d = tf.placeholder(tf.float32)         # [1] dimension

u = tf.tensordot(x,w, axes=1 ) + b
y = amp*tf.sigmoid(u)+offset

    # Error
loss = tf.square(d-y)

    # Optimize
    
grad_w, grad_b = tf.gradients(loss,[w,b])

w_new = w.assign(w - lr*grad_w)
b_new = b.assign(b - lr*grad_b)



## Loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

w_, b_ = sess.run([w,b])
print('w: {},\n b: {}\n'.format(w_,b_))

    # Containers for plotting
mse = []
idx = np.arange(len(X))
    # epoke loope
for i in range(iters):

    np.random.shuffle(idx)
    X,Y = X[idx], Y[idx]

    err_  = []
    for p in range(len(X)):
        sess.run([w_new, b_new], {x:X[p], d:Y[p]})
        err_.append(sess.run(loss, {x:X[p], d:Y[p]} ))

    mse.append(np.mean(err_))
        
        # Print the square error
    if (i%100==0):
        print('iter: %d, mse: %g'%(i, mse[i]))

    # plot the data
plt.figure(1)
plt.plot(np.arange(iters), mse)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.show()

    
    # Save values for plotting
w_, b_ = sess.run([w, b])
print('w: {}, b: {}'.format(w_, b_))
print('mse = %g'%mse[iters-1])

y_ = []
for p in np.arange(len(X)):
    y_.append( sess.run(y, {x:X[p]}) )
    
        

    

## Plot the prediction
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
ax.scatter(X[:,0], X[:,1], y_)
ax.set_title('Targets and Predictions')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.show()


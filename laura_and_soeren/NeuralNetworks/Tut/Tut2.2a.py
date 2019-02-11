## Design a GD perceptron to approximate the function y = 0.5 + x_1 + 4x_2^2 
import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## Generate training data
iters = 5000                        # Epokes
lr = tf.Variable(0.01, tf.float32)  # learning rate
no_data = 11*11

SEED = 10
np.random.seed(SEED)

    # Data is the grid from 0 < x < 1 med 0.1 interval = 11*11= 121 data poits
X = np.zeros((no_data,2))
Y = np.zeros((no_data,1))
i = 0

for x1 in np.arange(0.,1.01,0.1):
    for x2 in np.arange(0.,1.01,0.1):
        X[i] = [x1,x2]
        Y[i] = 0.5 + x1 + 4*x2**2
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

x = tf.placeholder(tf.float32, [None,2])    # [?,2] dimension
d = tf.placeholder(tf.float32, [None,1])    # [?,1] dimension

u = tf.matmul(x,w) + b
y = amp*tf.sigmoid(u)+offset

    # Error
loss = tf.reduce_mean(tf.square(d-y))

    # Optimize
    
grad_w, grad_b = tf.gradients(loss,[w,b])

#grad_w = tf.gradients(loss,[w])
#grad_b = tf.gradients(loss,[b])

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

    # epoke loope
for i in range(iters):
    
    w_, b_ = sess.run([w_new, b_new], {x:X, d:Y})
    mse.append(sess.run(loss, {x:X, d:Y} ))

        # Print the square error
    if (i%500==0):
        print('iter: %d, mse: %g'%(i, mse[i]))

    # Save values for plotting
w_, b_, y_, loss_ = sess.run([w, b, y, loss], {x:X, d:Y})
    
        
    # plot the data
plt.figure()
plt.plot(np.arange(iters), mse)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.show()
    

## Plot the prediction
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
ax.scatter(X[:,0], X[:,1], y_)
ax.set_title('Targets and Predictions')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.show()


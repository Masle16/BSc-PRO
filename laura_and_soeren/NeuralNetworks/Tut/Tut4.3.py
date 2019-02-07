## Design a perceptron layer to perform the following mapping with SGD

import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIVLE_DEVICES"]="1"

if not os.path.isdir('figures'):
    print('Creating a figure folder')
    os.makedirs('figures')


## Define the learning data and parameter
SEED = 10
tf.set_random_seed(SEED)
np.random.seed(SEED)
    
lr = 0.05
iters = 2000
num_data = 8
num_input = 2
num_output = 2

X_train = np.array([
    [0.50, 0.23],
    [0.20, 0.76],
    [0.17, 0.09],
    [0.69, 0.95],
    [0.00, 0.51],
    [0.81, 0.61],
    [0.72, 0.29],
    [0.92, 0.72]])
Y_train = np.array([
    [0.16, 0.74],
    [0.49, 0.97],
    [0.01, 0.26],
    [1.19, 1.70],
    [0.13, 0.52],
    [0.77, 1.48],
    [0.40, 1.04],
    [1.14, 1.70]])

## Build the graph
w = tf.Variable(tf.truncated_normal([num_input, num_output],stddev=1.0 / np.sqrt(num_input)))
b = tf.Variable(np.zeros(num_output), dtype = tf.float32)

x = tf.placeholder( dtype = tf.float32 )
d = tf.placeholder( dtype = tf.float32 )

u = tf.tensordot( tf.transpose(w) ,x , axes=1) + b     # u = w^T * x 
y = 2.0*tf.sigmoid(u)

loss = tf.reduce_sum(tf.square(d-y))

## Optimize
dy      = y*(1 - y/2.0)
grad_u  = -(d-y)*dy
grad_w  = tf.tensordot(x, tf.transpose( grad_u ), axes = 0 )

w_new   = w.assign( w - lr*grad_w ) 
b_new   = b.assign( b - lr*grad_u )


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
w_,b_   = sess.run( (w,b) )
print('w: {}, b: {}'.format(w_,b_))



#y_,loss_   = sess.run( (y, loss), {x:X_train[0], d:Y_train[0]} )
#print('y: {}\nloss: {}'.format(y_,loss_))

ERR_ = []
idx = np.arange(len(X_train))

## Loop
for i in range(iters):
    err_ = []
    np.random.shuffle( idx )
    X_train, Y_train = X_train[idx], Y_train[idx]
    
    for p in range( len(X_train) ):
        w_, b_ = sess.run( (w_new, b_new), {x:X_train[p], d:Y_train[p]} )
        err_.append( sess.run( loss, {x:X_train[p], d:Y_train[p]} ))
        if i == 0:
            print('iter: %d, p: %d, loss: %g'%(i,p,err_[p]))

    ERR_.append( np.mean(err_) )

    if i%100 == 0:
        print('iter: %d,  loss: %g'%(i,ERR_[i]))

        
## plot

w_,b_   = sess.run( (w,b) )
print('Weights and bias \nw: {}, b: {}'.format(w_,b_))
print('Error: %g'%(ERR_[iters-1]))

    # Predictions
pred = []
for p in range(len(X_train)):
    pred.append(sess.run( y, {x:X_train[p], d:Y_train[p]} ))

pred = np.array(pred)


plt.figure(1)
plt.plot( np.arange(iters), ERR_ )
plt.ylabel('error')
plt.xlabel('Iters')
plt.title('Error vs iters')

plt.figure(2)
plt.scatter( Y_train[:,0], Y_train[:,1])
plt.scatter( pred[:,0], pred[:,1])
plt.ylabel('y1')
plt.xlabel('y2')
plt.title('Predictions')

plt.show()
    

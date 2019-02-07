## USe GD on a logistic regression neuro to do the classification
import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    
## Learning data
lr = 0.4
iters = 1000
SEED = 10

np.random.seed(SEED)

inputs = np.array([
     [-1.75, 0.34, 1.15],
     [-0.25, 0.98, 0.51],
     [0.22, -1.07, -0.19],
     [0.26, -0.46, 0.44],
     [-0.58, 0.82, 0.67],
     [-0.1, -0.53, 1.03],
     [-0.44, -1.12, 1.62],
     [1.54, -0.25, -0.84],
     [0.18, 0.94, 0.73],
     [1.36, -0.33, 0.06]])

outputs = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 0]).reshape(10,1)

    # Plot the data
##fig = plt.figure(1)
##ax = fig.add_subplot(111, projection='3d')
##
##c1 = ax.scatter(inputs[outputs[:,0]==0,0],inputs[outputs[:,0]==0,1],inputs[outputs[:,0]==0,2],marker='x', label='class A')
##c2 = ax.scatter(inputs[outputs[:,0]==1,0],inputs[outputs[:,0]==1,1],inputs[outputs[:,0]==1,2],marker='x', label='class B')
##ax.set_xlabel('X Label')
##ax.set_ylabel('Y Label')
##ax.set_zlabel('Z Label')
#plt.show()

## Build the graph
x = tf.placeholder(tf.float32, inputs.shape )
d = tf.placeholder(tf.int32, outputs.shape )

w = tf.Variable( np.random.rand(3,1), dtype=tf.float32)
b = tf.Variable( 0., dtype=tf.float32 )

u = tf.matmul(x,w) + b
y = tf.sigmoid(u)

d_float = tf.cast(d, tf.float32)


#loss = - tf.reduce_sum( d_float*tf.log(f_u) + (1-d_float)*tf.log(1-f_u))
loss = - tf.reduce_sum( d_float*tf.log(y)   + (1-d_float)*tf.log(1-y) )
            # Checks the class error 
class_err = tf.reduce_sum( tf.cast( tf.not_equal( y > 0.5, outputs ), tf.int32))


    # Optimize
grad_u = -(d_float-y)
grad_w = tf.matmul(tf.transpose(x), grad_u )
grad_b = tf.reduce_sum( grad_u )

w_new = w.assign( w - lr*grad_w )
b_new = b.assign( b - lr*grad_b )


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

w_, b_ = sess.run([w,b])
print( 'w: {}, b: {}\n'.format(w_,b_) )

err = []
c_err = []

## Do the loop
for i in range(iters):
    
    w_, b_ = sess.run([w_new, b_new], {x:inputs, d:outputs})

    err.append( sess.run( loss, {x:inputs, d:outputs}) )
    c_err.append( sess.run( class_err, {x:inputs, d:outputs}) )
    if i%100 == 0:
        print('Iteration: %d'%(i))
        print('w: {}, b: {}'.format(w_, b_))
        print('Class error: %g'%c_err[i])
        print('Loss: %g\n'%err[i])
    


plt.figure(2)
plt.plot(np.arange(iters), err)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Iterations vs error')

plt.figure(3)
plt.plot(np.arange(iters), c_err)
plt.xlabel('Iterations')
plt.ylabel('Classe Error')
plt.title('Iterations vs class error')


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
X = np.arange(-1, 2, 0.1)
Y = np.arange(-1, 2, 0.1)
X,Y = np.meshgrid(X,Y)
Z = -(w_[0]*X + w_[1]*Y + b_)/w_[2]
decision_boundary = ax.plot_surface(X,Y,Z)

c1 = ax.scatter(inputs[outputs[:,0]==0,0],inputs[outputs[:,0]==0,1],inputs[outputs[:,0]==0,2],marker='x', label='class A')
c2 = ax.scatter(inputs[outputs[:,0]==1,0],inputs[outputs[:,0]==1,1],inputs[outputs[:,0]==1,2],marker='x', label='class B')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()
## Plot the learning curve
## Plot the error


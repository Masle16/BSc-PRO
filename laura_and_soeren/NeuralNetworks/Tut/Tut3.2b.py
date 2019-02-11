## Design a classifier for a dichotomizer using discrete logistis 
import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## Learning data
lr = 0.1
iters = 300

SEED = 100
np.random.seed(SEED)

X = np.array([
    [0.8, 0.5, 0.0],
    [0.9, 0.7, 0.3],
    [1.0, 0.8, 0.5],
    [0.0, 0.2, 0.3],
    [0.2, 0.3, 0.5],
    [0.4, 0.7, 0.8] ])
Y = np.array([0, 0, 0, 1, 1, 1])

print(X)
print(Y)
print(lr)

##fig = plt.figure(1)
##ax = fig.add_subplot(111, projection ='3d')
##ax.scatter(X[:,0], X[:,1])
##plt.xlabel('X')
##plt.ylabel('Y')
##plt.title('Training data')

## Build the graph
x = tf.placeholder( tf.float32 )
d = tf.placeholder( tf.float32 )

w = tf.Variable( np.random.rand(3), dtype = tf.float32 )
b = tf.Variable( 0., dtype = tf.float32 )

u = tf.tensordot( x, w, axes = 1) + b
y = 1 / (1 + tf.exp(-u))

delta = d - y

    # Optimize
w_new = w.assign( w + lr*delta*x )
b_new = b.assign( b + lr*delta )

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run( init )

w_, b_= sess.run([w,b])
print( 'w: {}, b: {} \n'.format(w_, b_))

idx = np.arange(len(X))
ERR = []

## Do the loop


for i in range(iters):
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]

    err_ = 0
    for p in range(len(X)):
        u_, y_, w_, b_ = sess.run([u, y, w_new, b_new], {x:X[p], d:Y[p]} )
        err_ += y_ != Y[p]

        if i == 0 :
            print('p: %d'%(p+1))
            print('w: {}, b: {}'.format(w_, b_))
            print('y: {}, d: {}'.format(y_,Y[p]))
        
    ERR.append(err_/len(X))
    print('epoch: %d, error: %g'%(i, ERR[i]))

print('w: {}, b: {}'.format(w_,b_))

# Plot learning curve
plt.figure(1)
plt.plot(range(iters),ERR)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.xticks(np.arange(iters//4)*4, np.arange(iters//4)*4)
plt.title('Iteration vs Error')
#plt.show()


#-----------------##

# plot data and the decision boundary
fig = plt.figure(2)
ax = fig.gca(projection='3d')
c1 = ax.scatter(X[Y==0,0],X[Y==0,1],X[Y==0,2],marker='x', label='class A')
c2 = ax.scatter(X[Y==1,0],X[Y==1,1],X[Y==1,2],marker='x', label='class B')
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



plt.show()


    



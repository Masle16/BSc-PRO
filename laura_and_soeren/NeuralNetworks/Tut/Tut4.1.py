## Design a softmax layer of neurons to perform the following classification using GD

import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"         # -1 = CPU, 1 = GPU

## Learning data
num_features = 2
num_classes = 3
num_data = 18

lr = 0.05
iters = 1500

SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)

       
X_train = np.array([[ 0, 4],
              [-1, 3],
              [ 2, 3],
              [-2, 2],
              [ 0, 2],
              [ 1, 2],
              [-1, 2],
              [-3, 1],
              [-1, 1],
              [ 2, 1],
              [ 4, 1],
              [-2, 0],
              [ 1, 0],
              [ 3, 0],
              [-3,-1],
              [-2,-1],
              [ 2,-1],
              [ 4,-1] ])

Y_train = np.array([0,0,0,1,1,0,1,1,1,2,2,1,2,2,1,1,2,2]).astype(int)

K_train = np.zeros((num_data, num_classes)).astype(float)
for p in range(num_data):
    K_train[p,Y_train[p]] = 1

print(X_train, K_train)

##plt.figure(1)
##plt.scatter( X_train[Y_train[:,0]==0,0], X_train[Y_train[:,0]==0,1], c='r', marker='o')
##plt.scatter( X_train[Y_train[:,0]==1,0], X_train[Y_train[:,0]==1,1], c='Y', marker='x')
##plt.scatter( X_train[Y_train[:,0]==2,0], X_train[Y_train[:,0]==2,1], c='B', marker='o')
##plt.xlabel('X')
##plt.ylabel('Y')
##plt.title('Training data')
##plt.show()

## Build the graph

X = tf.placeholder( tf.float32, X_train.shape)
K = tf.placeholder( tf.float32, K_train.shape)

    # Generate random distributed W
W = tf.Variable(tf.truncated_normal([2, 3],stddev=1.0 / np.sqrt(4)))
    # Generate b. Only need to be one row, becuase tf will do the rest. b^T= [x x x]
b = tf.Variable( np.zeros( (num_classes)), dtype=tf.float32)        

U = tf.matmul(X,W) + b  
p = tf.exp(U)/tf.reduce_sum(tf.exp(U), axis=1, keepdims=True)    # Predictions, f(u)    
Y = tf.argmax(p,1)                                          # Get the predicted labels

p_sum = tf.reduce_sum(p, 1, keepdims=True)


        # tf.not_equal is used to find the number of missmatch
err = tf.reduce_sum( tf.cast( tf.not_equal( tf.argmax(K, 1), Y), tf.int32 ) )
loss = - tf.reduce_sum(tf.log(p)*K)                           # Calculate the entropy


## Optimze parameters
grad_u = -(K-p)
grad_w = tf.matmul( tf.transpose(X), grad_u )
grad_b = tf.reduce_sum(grad_u, axis = 0)

w_new = W.assign( W - lr*grad_w )
b_new = b.assign( b - lr*grad_b )


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

w_,b_= sess.run( [W, b] )
print('w: {},\n b: {}\n'.format(w_, b_))


ERR_ = []
entropy_ = []
## Do the loop
for i in range(iters):
    
    if i == 0:
        u_, p_, y_, l_, e_, du_ = sess.run([U, p, Y, loss, err, grad_u], {X: X_train, K:K_train})
        print('iter: {}'.format(i+1))
        print('u: {}'.format(u_))
        print('p: {}'.format(p_))
        print('y: {}'.format(y_))
        print('grad_u: {}'.format(du_))
        print('loss: {}'.format(l_))
        print('error: {}'.format(e_))

    sess.run([w_new, b_new], {X:X_train, K:K_train})
    ERR_.append(sess.run(err, {X:X_train, K:K_train}))
    entropy_.append(sess.run(loss, {X:X_train, K:K_train}))

    if not i%100:
        print('epoch: %d, loss: %g, error: %d'%(i,entropy_[i], ERR_[i]))

## Training accuracy
w_, b_, p_, y_, l_, e_ = sess.run([W, b, p, Y, loss, err], {X:X_train, K:K_train})
print("w: %s, b: %s"%(w_, b_))
print("p: %s"%p_)
print("y: %s"%y_)
print("loss: %g, error: %g"%(l_, e_))
        
##
##    if i == iters -1:
##        print( sess.run(p, {X:X_train, K:K_train}) )
##
#### Plot the error and decision boundary 
##
plt.figure(2)
plt.plot(range(iters), ERR_)
plt.xlabel('Interation')
plt.ylabel('Errors')
plt.title('Iteration vs errors')

plt.figure(3)
plt.plot(range(iters),entropy_)
plt.xlabel('ITerations')
plt.ylabel('Entropy')
plt.title('Iteration vs Entropy')

plt.show()



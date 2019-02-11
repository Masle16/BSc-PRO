## Use mini-batch gradient decent learning to train a softmax layer to classify Iris dataset
import tensorflow as tf
import numpy as np
import pylab as plt
import os

from sklearn import datasets
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

    ## Make dir for pictures
if not os.path.isdir('figures'):
    print('Creating the figures folder')
    os.makedirs('figures')

## Manipulate the training data
    ## Learning data and parameters
iris = datasets.load_iris() # Import dataset
iris_data = iris.data       # assign data
iris_target = iris.target   # assign target

    ## Setup learning parameters
lr = 0.01
iters = 2000
batch_size = 16
num_features = 4
num_classes = 3
num_train_data = 130
num_test_data = 20
mini_batchs = np.floor(num_train_data/batch_size).astype(int)

    ## Generate random seeds
SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)

    ## Shuffle the dataset, because it is ordered
idx = np.arange(len(iris.data))
np.random.shuffle(idx)
iris_data, iris_target = iris_data[idx], iris_target[idx]

    ## Take out data for test (20) and training (130)
X_train = iris_data[ :num_train_data, : ]
Y_train = iris_target[ :num_train_data ]

X_test = iris_data[ num_train_data:, : ]
Y_test = iris_target[ num_train_data: ]

    ## Generate the K matrix
K_train = np.zeros( (num_train_data, num_classes) ).astype(int)
for p in range(num_train_data):
    K_train[ p, Y_train[p] ] = 1

    ## Generate the K matrix
K_test = np.zeros( (num_test_data, num_classes) ).astype(int)
for p in range(num_test_data):
    K_test[ p, Y_test[p] ] = 1

## Build the graph
w = tf.Variable( tf.truncated_normal( (num_features, num_classes), stddev=1.0 / np.sqrt(4) ) )
b = tf.Variable( np.zeros(num_classes), dtype = tf.float32 )

x = tf.placeholder( dtype = tf.float32, shape = [batch_size, num_features] )
k = tf.placeholder( dtype = tf.float32, shape = [batch_size, num_classes] )

u = tf.matmul( x,w ) + b
p = tf.exp(u)/tf.reduce_sum( tf.exp(u), axis = 1, keepdims = True )
y = tf.argmax(p, axis = 1)

error = tf.reduce_sum( tf.cast( tf.not_equal( tf.argmax(k,1), y ), dtype = tf.int32 ) )
loss = - tf.reduce_sum( tf.log(p)*k )       # Times k becuase we want to remove the wrong ones


    ## Optimize parameters
grad_u = - (k - p)
grad_w = tf.matmul( tf.transpose(x), grad_u )
grad_b = tf.reduce_sum( grad_u, axis = 0 )

w_new = w.assign( w - lr*grad_w )
b_new = b.assign( b - lr*grad_b )

    ## initialize and print w and b
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

w_,b_ = sess.run([w,b])
print('w: {}, b: {}'.format(w_,b_))

## Do the batch loop
## We need to take the the batch multiple time and do it iters times. it need to be
## shuffled every time we do a epoke. we need to find the error over each epoke and the loss

err = []        # Container for the amount of errors per epoke
entropy = []    # Container for the the entropy per epoke

for i in range(iters):

    np.random.shuffle(idx)
    iris_data, iris_target = iris_data[idx], iris_target[idx]
    
    err_ = []       # Container for the amount of error per batch
    entropy_ = []   # Container for the entropy per batch


    for q in range( mini_batchs ):
            # Define learning batch
        interval_min = q*batch_size
        interval_max = (q+1)*batch_size

            # Print first iteration           
        if i == 0:
            u_,p_,y_,e_,l_,w_,b_ = sess.run( [u,p,y,error,loss,w,b], {x:X_train[interval_min:interval_max], k:K_train[interval_min:interval_max]})
            print('p: %d'%q)
            print('w: {} \nb: {}'.format(w_,b_))
            print('u: {} \np: {}'.format(u_,p_))
            print('y: {}'.format(y_))
            print('Error: {} \nLoss: {}'.format(e_,l_))

            # Safe error values
        err_.append( sess.run( error, {x:X_train[interval_min:interval_max], k:K_train[interval_min:interval_max]}))
        entropy_.append( sess.run( loss, {x:X_train[interval_min:interval_max], k:K_train[interval_min:interval_max]}))
            # Update w and b
        sess.run( [w_new,b_new], {x:X_train[interval_min:interval_max], k:K_train[interval_min:interval_max]})
        
        # Take the mean of the error arrayes
    err.append( np.mean(err_) )
    entropy.append( np.mean(entropy_) )

        # Print epoke values
    if i%100 == 0:
        print('Epoch: %d, Loss: %g, Errors: %d'%(i, entropy[i], err[i]))
    
        
## print the learned parameters
w_, b_ = sess.run( [w,b] )
print('\nw: {} \nb: {}'.format(w_,b_))     

## plot the curves
plt.figure(1)
plt.plot(range(iters), err)
plt.xlabel('Iteration')
plt.ylabel('Errors')
plt.title('Iteration vs Errors')
plt.savefig('./figures/t4q2_1-b16std.png')

plt.figure(2)
plt.plot(range(iters), entropy)
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.title('Iteration vs Entropy')
plt.savefig('./figures/t4q2_2_b16std.png')

p_,y_,e_ = sess.run( [p,y,error], {x:X_test[:batch_size], k:K_test[:batch_size]} )

print('Prediction: {}\n Labels: {}\nError: {}'.format(p_,y_,e_))
print(Y_test[batch_size])
print(np.not_equal(y_,Y_test[:batch_size])) # True if there is an wrong classification

plt.show()


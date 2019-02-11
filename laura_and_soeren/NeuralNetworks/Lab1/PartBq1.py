#
# Project 1, part B question 1
# 

import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('Figures_B1'):
    print('creating the figures folder')
    os.makedirs('Figures_B1')

    # Assgin GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## Define the constant
NUM_FEATURES = 8
NUM_NEURONS = 30

LR = 10**(-7)
BETA = 10**(-3)
EPOCHS = 600
TEST_PLOT = 50
BATCH_SIZE = 32
SEED = 10
np.random.seed(SEED)

## Import working data
        # Read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

        # Shuffle the data
idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

        # Devide in validation and test data
m = 3* X_data.shape[0] // 10
trainX, trainY = X_data[m:], Y_data[m:]
testX, testY = X_data[:m], Y_data[:m]

        # Experiment with small datasets
#trainX = trainX[:1000]
#trainY = trainY[:1000]

        # Convert the training data to std
trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
testX = (testX - np.mean(trainX, axis=0))/ np.std(trainX, axis=0)


## Build the dataflow graph
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

        # Hidden layer, ReLu
w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_NEURONS],
                                     stddev=1.0 / np.sqrt(NUM_FEATURES),
                                     dtype=tf.float32),
                 name='weights')
b1 = tf.Variable(tf.zeros([NUM_NEURONS]), dtype=tf.float32, name='biases')
h = tf.nn.relu( tf.matmul(x, w1) + b1 )

        # Output layer, Linear
w2 = tf.Variable(tf.truncated_normal([NUM_NEURONS, 1],
                                     stddev=1.0 / np.sqrt(NUM_NEURONS),
                                     dtype=tf.float32),
                 name='weights')
b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
y = tf.matmul(h, w2) + b2 

        # Loss with L2 regulation
entropy = tf.reduce_mean( tf.square(y_ - y))
L2 = BETA*( tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) )
loss = entropy + L2

        # Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer( LR )
train_op = optimizer.minimize( loss )

## Start the session and train the network
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        idx = np.arange(trainX.shape[0])
                # Error containers for training end test error
        tr_err, te_err = [],[]
        
        for i in range( EPOCHS ):
                        # Shuffle data for mini batch                
                np.random.shuffle(idx)

                trainX, trainY = trainX[idx], trainY[idx]
                
                for p in range(trainX.shape[0]// BATCH_SIZE ):
                        train_op.run(feed_dict=
                                     {x: trainX[ p*BATCH_SIZE: (p+1)*BATCH_SIZE],
                                      y_: trainY[ p*BATCH_SIZE: (p+1)*BATCH_SIZE]})


                tr_err.append( entropy.eval(feed_dict={x: trainX, y_: trainY}))
                te_err.append( entropy.eval(feed_dict={x: testX,  y_: testY }) )
                if i % 100 == 0:
                        print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))


        val_predictions = y.eval( feed_dict = {x: trainX[:TEST_PLOT]} )
        test_predictions = y.eval( feed_dict = {x: testX[:TEST_PLOT]} )


## (a) Plot learning curves
plt.figure(1)
plt.plot(range( EPOCHS ), tr_err, label = 'train error')
plt.xlabel('Epochs')
plt.ylabel('Mean square error')
plt.title('Validation error of 32 mini batch learning')
plt.legend()
plt.savefig('Figures_B1/trainingError.png')

plt.figure(2)
plt.plot(range( EPOCHS ), te_err, label = 'test error')
plt.xlabel('Epochs')
plt.ylabel('Mean square error')
plt.title('Test error of 32 mini batch learning')
plt.legend()
plt.savefig('Figures_B1/testError.png')



## (b) Plot the prediction value and target values for any 50 test samples

targets = np.asarray( testY[:TEST_PLOT] )
targets = np.squeeze( targets )
val_predictions = np.squeeze( val_predictions )
test_predictions = np.squeeze( test_predictions )

targets = np.sort( targets )
val_predictions = np.sort( val_predictions )
test_predictions = np.sort( test_predictions )


fig = plt.figure(3)
ax1 = fig.add_subplot(111)

ax1.scatter( range( TEST_PLOT ), val_predictions, marker="x", label='Pred')
ax1.scatter( range( TEST_PLOT ), targets, marker="o", label='Targets')
plt.legend(loc='upper left');

plt.xlabel('Epochs')
plt.ylabel('Targets/val_Predictions')
plt.title('validation predictions / targets vs Epochs')
plt.savefig('Figures_B1/val_predictions.png')


fig = plt.figure(4)
ax1 = fig.add_subplot(111)

ax1.scatter( range( TEST_PLOT ), test_predictions, marker="x", label='Pred')
ax1.scatter( range( TEST_PLOT ), targets, marker="o", label='Targets')
plt.legend(loc='upper left');

plt.xlabel('Epochs')
plt.ylabel('Targets/val_Predictions')
plt.title('test predictions / targets vs Epochs')
plt.savefig('Figures_B1/test_predictions.png')

plt.show()


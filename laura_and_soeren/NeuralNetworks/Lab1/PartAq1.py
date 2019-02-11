#
# Project 1, part A, question 1
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

import os
if not os.path.isdir('Figures_A1'):
    print('creating the figures folder')
    os.makedirs('Figures_A1')

# Scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

## Define the constants
NUM_FEATURES = 36
NUM_NEURONS = 10
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 5000
batch_size = 32
beta = 10**(-6)

TEST_PLOT = 50

seed = 10
np.random.seed(seed)

## Import train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)

    # Experiment with small datasets
#trainX = trainX[:100]
#train_Y = train_Y[:100]
    
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0)) #scaling of input
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix, targets represented as so

## Import test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

testX = scale(testX, np.min(trainX, axis=0), np.max(trainX, axis=0)) #scaling of test input with train data
test_Y[test_Y == 7] = 6

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix

## Build the graph for the deep net
    # Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])

    # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Hidden perceptron layer
w1 = tf.Variable(
    tf.truncated_normal([NUM_FEATURES, NUM_NEURONS],
                        stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
    name='weights')
b1 = tf.Variable(tf.zeros([NUM_NEURONS]),name='biases')
z1 = tf.matmul(x, w1) + b1
h1 = tf.nn.relu(z1)

    # Output softmax layer
w2 = tf.Variable(
    tf.truncated_normal([NUM_NEURONS, NUM_CLASSES],
                        stddev=1.0 / math.sqrt(float(NUM_NEURONS))),
    name='weights')
b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
logits  = tf.matmul(h1, w2) + b2

    # Loss with L2 regularization
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y_, logits=logits) )
regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
loss = cross_entropy + beta*regularization

    # Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

    # Where should I put these?
correct_prediction = tf.cast(
    tf.equal( tf.argmax(logits, 1), tf.argmax(y_, 1) ), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)


## Start session and train network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    idx = np.arange(trainX.shape[0])
    te_acc, tr_err = [],[] # Container for accuracies
    
    for i in range(epochs):
            # Shuffle the data for mini batch
        np.random.shuffle(idx)

        trainX, trainY = trainX[idx], trainY[idx]

        for p in range(trainX.shape[0]// batch_size):
            train_op.run(feed_dict={x: trainX[p*batch_size: (p+1)*batch_size],
                                    y_: trainY[p*batch_size: (p+1)*batch_size]})

        tr_err.append( cross_entropy.eval(feed_dict={x: trainX, y_: trainY}))
        te_acc.append( accuracy.eval(feed_dict={x:testX, y_: testY}))
        print (correct_prediction.eval(feed_dict={x:testX, y_: testY}))
        print(logits.eval(feed_dict={x:testX, y_: testY}))
        print('iter %d: train error: %g test accuracy: %g' % (i, tr_err[i], te_acc[i]))



        val_predictions = logits.eval( feed_dict = {x: trainX[:TEST_PLOT]} )
        test_predictions = logits.eval( feed_dict = {x: testX[:TEST_PLOT]} )

## Plot learning curves
plt.figure(1)
plt.plot(range(epochs), tr_err)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Training error')
plt.title('A1: Training error vs epochs')
plt.savefig('Figures_A1/trainErr.png')


plt.figure(2)
plt.plot(range(epochs), te_acc)
plt.xlabel(str(epochs) + 'iterations')
plt.ylabel('Test accuracy')
plt.title('A1: Test accuracy vs epochs')
plt.savefig('Figures_A1/testAcc.png')


plt.show()


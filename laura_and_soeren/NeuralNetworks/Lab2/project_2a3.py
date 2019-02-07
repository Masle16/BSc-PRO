#
# Project 2, Part A, Question 2
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import multiprocessing as mp

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

######################### Define Constants #########################

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 128
F_SIZE_1 = 50
F_SIZE_2 = 250
GAMMA = 0.1
KEEP_PROB = 0.9


SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)

######################### Read in the data #########################

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_

############ Build the convolutional neural network ################

def cnn(images, dropout):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS]) #(BATCH_SIZE, 32,32,3)
    
    # Conv 1 - maps one color image to 50 feature maps.  output: [BATCH_SIZE, 24,24,filter-size1]
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, F_SIZE_1], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([F_SIZE_1]), name='biases_1')

    if dropout == 1:
        conv_1 = tf.nn.dropout((tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)), KEEP_PROB)
    elif dropout == 0:
        conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)


    # First pooling layer - downsamples by 2x.  output: [BATCH_SIZE, 12, 12, filter-size1]
    pool_1 = tf.nn.max_pool(conv_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool_1')

    # Conv 2 - maps 'filter-size1' feature map to filter-size2. output: [BATCH_SIZE, 8, 8, filter-size2]
    W2 = tf.Variable(tf.truncated_normal([5, 5, F_SIZE_1, F_SIZE_2], stddev=1.0/np.sqrt(NUM_CHANNELS*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([F_SIZE_2]), name='biases_2')

    if dropout == 1: 
        conv_2 = tf.nn.dropout((tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)), KEEP_PROB)
    elif dropout == 0: 
        conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)

    # Second pooling layer - downsamples by 2x. output: [BATCH_SIZE, 4, 4, filter-size2] 
    pool_2 = tf.nn.max_pool(conv_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool_2')
        
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim]) # output: [BATCH_SIZE, 960]


    # Fully connected layer - after 2 rounds of downsampling our 32x32 image
    # is down to 4x4xfilter-size2 feature maps -- maps this to 300 features
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3') 

    if dropout == 1:    
        h_fc = tf.nn.dropout((tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)), KEEP_PROB)
    elif dropout == 0:
        h_fc = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)

    # Softmax output layer
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')

    if dropout == 1:
        logits = tf.nn.dropout((tf.matmul(h_fc, W4) + b4), KEEP_PROB)
    elif dropout == 0:
        logits = tf.matmul(h_fc, W4) + b4

    return conv_1, pool_1, conv_2, pool_2, logits

########################### Main Function #############################

def main():

    # Read in the training and test data
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    trainX, trainY = trainX[0:1000], trainY[0:1000]
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    # Scale the data - should test data be scaled as well?
    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

##### Run the network using different filter sizes #####
    
    # Run the deep net
    conv_1, pool_1, conv_2, pool_2, logits = cnn(x, 0)

    # Define loss, accuracy and optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    
    # Train the network using different optimizer algorithms 
    train_step1 = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    train_step2 = tf.train.MomentumOptimizer(LEARNING_RATE, GAMMA).minimize(loss)
    train_step3 = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
    train_step4 = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    correct_predictions = tf.equal(tf.argmax(y_,1), tf.argmax(logits,1))
    correct_predictions = tf.cast(correct_predictions, tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)

    N = len(trainX)
    idx = np.arange(N)
    
    # Train the network using mini-batch gradient descent, and use it on test data to evaluate performance
    with tf.Session() as sess:

        plt.figure(1)
        plt.figure(2)

        print('gd:')
        sess.run(tf.global_variables_initializer())
        
        train_err, test_acc = [], []
        for e in range(EPOCHS):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                train_step1.run(feed_dict= {x: trainX[start:end], y_: trainY[start:end]})

            train_err.append(loss.eval(feed_dict= {x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict= {x: testX, y_: testY}))
            print('epoch', e, 'entropy', train_err[e], 'test accuracy', test_acc[e])

        plt.figure(1)
        plt.plot(np.arange(EPOCHS), test_acc, label='gd')
        plt.figure(2)
        plt.plot(np.arange(EPOCHS), train_err, label='gd')


        print('momentum:')
        sess.run(tf.global_variables_initializer())

        train_err, test_acc = [], []
        for e in range(EPOCHS):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                train_step2.run(feed_dict= {x: trainX[start:end], y_: trainY[start:end]})

            train_err.append(loss.eval(feed_dict= {x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict= {x: testX, y_: testY}))
            print('epoch', e, 'entropy', train_err[e], 'test accuracy', test_acc[e])

        plt.figure(1)
        plt.plot(np.arange(EPOCHS), test_acc, label='momentum')
        plt.figure(2)
        plt.plot(np.arange(EPOCHS), train_err, label='momentum')

        print('RMSProp:')
        sess.run(tf.global_variables_initializer())

        train_err, test_acc = [], []
        for e in range(EPOCHS):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                train_step3.run(feed_dict= {x: trainX[start:end], y_: trainY[start:end]})

            train_err.append(loss.eval(feed_dict= {x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict= {x: testX, y_: testY}))
            print('epoch', e, 'entropy', train_err[e], 'test accuracy', test_acc[e])

        plt.figure(1)
        plt.plot(np.arange(EPOCHS), test_acc, label='RMSProp')
        plt.figure(2)
        plt.plot(np.arange(EPOCHS), train_err, label='RMSProp')

        print('Adam:')
        sess.run(tf.global_variables_initializer())

        train_err, test_acc = [], []
        for e in range(EPOCHS):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                train_step4.run(feed_dict= {x: trainX[start:end], y_: trainY[start:end]})

            train_err.append(loss.eval(feed_dict= {x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict= {x: testX, y_: testY}))
            print('epoch', e, 'entropy', train_err[e], 'test accuracy', test_acc[e])

        plt.figure(1)
        plt.plot(np.arange(EPOCHS), test_acc, label='Adam')
        plt.figure(2)
        plt.plot(np.arange(EPOCHS), train_err, label='Adam')

        plt.figure(1)
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')
        plt.title('Test Accuracies with Different Optimizers.')
        plt.legend(loc='lower right')
        plt.savefig('./figures/opt-accs.png')

        plt.figure(2)
        plt.xlabel('epochs')
        plt.ylabel('train error')
        plt.title('Train Errors with Different Optimizers.')
        plt.legend(loc='lower right')
        plt.savefig('./figures/opt-errs.png')

        plt.show()

if __name__ == '__main__':
  main()

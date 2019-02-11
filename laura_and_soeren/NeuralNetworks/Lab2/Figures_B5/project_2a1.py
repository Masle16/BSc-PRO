#
# Project 2, Part A, Question 1
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle

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
EPOCHS = 5000
BATCH_SIZE= 128

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

def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS]) #(BATCH_SIZE, 32,32,3)
    
    # Conv 1 - maps one color image to 50 feature maps.  output: [BATCH_SIZE, 24,24,50]
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')
    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)


    # First pooling layer - downsamples by 2x.  output: [BATCH_SIZE, 12, 12, 50]
    pool_1 = tf.nn.max_pool(conv_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool_1')
    

    # Conv 2 - maps 50 feature map to 60. output: [BATCH_SIZE, 8, 8, 60]
    W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(NUM_CHANNELS*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([60]), name='biases_2')
    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)


    # Second pooling layer - downsamples by 2x. output: [BATCH_SIZE, 4, 4, 60] 
    pool_2 = tf.nn.max_pool(conv_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID', name='pool_2')

    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim]) # output: [BATCH_SIZE, 960]


    # Fully connected layer - after 2 rounds of downsampling our 32x32 image
    # is down to 4x4x60 feature maps -- maps this to 300 features
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3') 
        
    h_fc = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)


    # Softmax output layer
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')

    logits = tf.matmul(h_fc, W4) + b4
    return conv_1, pool_1, conv_2, pool_2, logits

########################### Main Function ###########################

def main():

    # Read in the training and test data
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    trainX, trainY = trainX[0:1000], trainY[0:1000]
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    # Scale the data - should test data be scaled as well?
    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    testX = (testX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Run the deep net
    conv_1, pool_1, conv_2, pool_2, logits = cnn(x)


    # Define loss, accuracy and optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    correct_predictions = tf.equal(tf.argmax(y_,1), tf.argmax(logits,1))
    correct_predictions = tf.cast(correct_predictions, tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)

    N = len(trainX)
    idx = np.arange(N)
    
    # Train the network using mini-batch gradient descent, and use it on test data to evaluate performance
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err, test_acc = [], []
        for e in range(EPOCHS):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                train_step.run(feed_dict= {x: trainX[start:end], y_: trainY[start:end]})

            train_err.append(loss.eval(feed_dict= {x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict= {x: testX, y_: testY}))
            print('epoch', e, 'entropy', train_err[e], 'test accuracy', test_acc[e])


############################ Plots ############################

        # Test accuracy
        plt.figure()
        plt.suptitle('Test Accuracy using Mini-Batch GD')
        plt.plot(np.arange(EPOCHS), test_acc, label='mini-batch gradient descent')
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')
        plt.legend(loc='lower right')
        plt.savefig('./figures/test-acc.png')

        # Training cost
        plt.figure()
        plt.suptitle('Training Cost using Mini-Batch GD')
        plt.plot(np.arange(EPOCHS), train_err, label='mini-batch gradient descent')
        plt.xlabel('epochs')
        plt.ylabel('training cost')
        plt.savefig('./figures/train-cost.png')

        # Take out two random test patterns from the data
        ind = np.random.randint(low=0, high=1000)
        X1 = trainX[ind,:]
        ind2 = np.random.randint(low=0, high=1000)
        X2 = trainX[ind2,:]

        # The 2 random input images taken out
        plt.figure()
        plt.suptitle('Input Image 1')
        plt.gray()
        plt.axis('off')
        plt.imshow(X1.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0))
        plt.savefig('./figures/input-img1.png')

        plt.figure()
        plt.suptitle('Input Image 2')
        plt.gray()
        plt.axis('off')
        plt.imshow(X2.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0))
        plt.savefig('./figures/input-img2.png')

        # Used for plotting the feature maps
        conv_1_1, pool_1_1, conv_2_1, pool_2_1 = sess.run([conv_1, pool_1, conv_2, pool_2], {x: X1.reshape(1, IMG_SIZE*IMG_SIZE*NUM_CHANNELS)})
        conv_1_2, pool_1_2, conv_2_2, pool_2_2 = sess.run([conv_1, pool_1, conv_2, pool_2], {x: X2.reshape(1, IMG_SIZE*IMG_SIZE*NUM_CHANNELS)})

        ##### Image 1 #####
        
        # Plot feature map at convolution layer 1, test pattern 1
        plt.figure()
        plt.suptitle('Feature Map at Convolution Layer 1, Image 1')
        plt.gray()
        conv_1_1 = np.array(conv_1_1)
        for i in range(50):
            plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(conv_1_1[0,:,:,i])
        plt.savefig('./figures/feature-map-conv1_1.png')
    
        # Plot feature map at pooling layer 1, test pattern 1
        plt.figure()
        plt.suptitle('Feature Map at Pooling Layer 1, Image 1')
        plt.gray()
        pool_1_1 = np.array(pool_1_1)
        for i in range(50):
            plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(pool_1_1[0,:,:,i ])
        plt.savefig('./figures/feature-map-pool1_1.png')
        
        # Plot feature map at convolution layer 2, test pattern 1
        plt.figure()
        plt.suptitle('Feature Map at Convolution Layer 2, Image 1')
        plt.gray()
        conv_2_1 = np.array(conv_2_1)
        for i in range(60):
            plt.subplot(10, 6, i+1); plt.axis('off'); plt.imshow(conv_2_1[0,:,:,i])
        plt.savefig('./figures/feature-map-conv2_1.png')
        
        # Plot feature map at pooling layer 2, test pattern 1
        plt.figure()
        plt.suptitle('Feature Map at Pooling Layer 2, Image 1')
        plt.gray()
        pool_2_1 = np.array(pool_2_1)
        for i in range(60):
            plt.subplot(10, 6, i+1); plt.axis('off'); plt.imshow(pool_2_1[0,:,:,i ])
        plt.savefig('./figures/feature-map-pool2_1.png')

        ##### Image 2 #####
    
        # Plot feature map at convolution layer 1, test pattern 2
        plt.figure()
        plt.suptitle('Feature Map at Convolution Layer 1, Image 2')
        plt.gray()
        conv_1_2 = np.array(conv_1_2)
        for i in range(50):
            plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(conv_1_2[0,:,:,i])
        plt.savefig('./figures/feature-map-conv1_2.png')
    
        # Plot feature map at pooling layer 1, test pattern 2
        plt.figure()
        plt.suptitle('Feature Map at Pooling Layer 1, Image 2')
        plt.gray()
        pool_1_2 = np.array(pool_1_2)
        for i in range(50):
            plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(pool_1_2[0,:,:,i ])
        plt.savefig('./figures/feature-map-pool1_2.png')
        
        # Plot feature map at convolution layer 2, test pattern 2
        plt.figure()
        plt.suptitle('Feature Map at Convolution Layer 2, Image 2')
        plt.gray()
        conv_2_2 = np.array(conv_2_2)
        for i in range(60):
            plt.subplot(10, 6, i+1); plt.axis('off'); plt.imshow(conv_2_2[0,:,:,i])
        plt.savefig('./figures/feature-map-conv2_2.png')
        
        # Plot feature map at pooling layer 2, test pattern 2
        plt.figure()
        plt.suptitle('Feature Map at Pooling Layer 2, Image 2')
        plt.gray()
        pool_2_2 = np.array(pool_2_2)
        for i in range(60):
            plt.subplot(10, 6, i+1); plt.axis('off'); plt.imshow(pool_2_2[0,:,:,i ])
        plt.savefig('./figures/feature-map-pool2_2.png')

        #plt.show()

if __name__ == '__main__':
  main()

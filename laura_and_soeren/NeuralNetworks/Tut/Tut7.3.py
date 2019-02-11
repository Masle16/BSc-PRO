# input 28x28, feature maps x2,
# 25  Filters of 9x9
# Polling window of 4x4 

import tensorflow as tf
import numpy as np
import pylab 
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


## Define parameters


BATCH_SIZE = 128
LR = 10**(-3)
EPOCHS = 500
SEED = 10

np.random.seed(SEED)
tf.set_random_seed(SEED)


def cnn(x):
        # input sixe 28x28
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolution layer - maps one grayscale image to 25 feature maps.
    W_conv = weight_variable([9, 9, 1, 25])
    b_conv = bias_variable([25])
    
    u_conv = tf.nn.conv2d(x_image, W_conv, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv
    h_conv = tf.nn.sigmoid(u_conv)

    # Poling layer  - downsamples by 4X. # input size 20x20 
    h_pool = tf.nn.max_pool(h_conv, ksize=[1, 4, 4, 1], strides=[1,4,4,1], padding= 'VALID')
        # Flatten output for DNN
    h_pool_flat = tf.reshape(h_pool, [-1, 5*5*25])
    
    # Fully connected layer 1   # input size 5x5 * 25
    W_fc = weight_variable([5* 5* 25, 10])
    b_fc = bias_variable([10])

    y_fc = tf.matmul(h_pool_flat, W_fc) + b_fc 
    
    return W_conv, h_conv, h_pool, y_fc

    


### Computational graph
    x = tf.palceholder(tf.float32, [None, 28, 28, 1])

    w = tf.Variable(W, tf.float32)
    b = tf.Variable(B, tf.float32)


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main():
        # Import data
    mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
    trainX, trainY  = mnist.train.images[:12000], mnist.train.labels[:12000]
    testX, testY = mnist.test.images[:2000], mnist.test.labels[:2000]

        # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

        # Run the deep net
    W_conv, h_conv, h_pool, y_fc = cnn(x)
    
        # Define loss and optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_,
                                                               logits = y_fc)
    cross_entropy = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer( LR ).minimize( cross_entropy )

    correct_predictions = tf.equal( tf.argmax(y_,1), tf.argmax(y_fc,1) )
    correct_predictions = tf.cast(correct_predictions, tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)

    N = len(trainX)
    idx = np.arange(N)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        for i in range(EPOCHS):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                train_step.run(feed_dict= {x: trainX[start:end], y_: trainY[start:end]})

            test_acc.append(accuracy.eval(feed_dict= {x: testX, y_: testY}))
            print('iter %d: test accuracy %g'%(i, test_acc[i]))



## Do the plotting
            # Plot the accuracy
        pylab.figure()
        pylab.plot(np.arange(EPOCHS), test_acc, label='gradient descent')
        pylab.xlabel('epochs')
        pylab.ylabel('test accuracy')
        pylab.legend(loc='lower right')

            # Plot the mapfeatures
        W_conv_ = sess.run(W_conv)
        W_conv_ = np.array(W_conv_)
        pylab.figure()
        pylab.gray()
        for i in range(25):
            pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(W_conv_[:,:,0,i])


            # Take random picture and print the feature maps
        ind = np.random.randint(low=0, high=55000)
        X = mnist.train.images[ind,:]

        pylab.figure()
        pylab.gray()
        pylab.axis('off'); pylab.imshow(X.reshape(28,28))  

        h_conv_, h_pool_ = sess.run([h_conv, h_pool], {x: X.reshape(1,784)})

        pylab.figure()
        pylab.gray()
        h_conv_ = np.array(h_conv_)
        for i in range(25):
            pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(h_conv_[0,:,:,i])

        pylab.figure()
        pylab.gray()
        h_pool_ = np.array(h_pool_)
        for i in range(25):
            pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(h_pool_[0,:,:,i])


        pylab.show()

if __name__ == '__main__':
    main()



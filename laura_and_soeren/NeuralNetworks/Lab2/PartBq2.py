import numpy as np
import pylab as plt
import pandas
import tensorflow as tf
import csv

import os
if not os.path.isdir('Figures_B2'):
    print('creating the figures folder')
    os.makedirs('Figures_B2')

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
EMBEDDING_SIZE = 20

BATCH_SIZE = 128
EPOCHS = 200
LR = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 100
tf.set_random_seed(seed)
np.random.seed(seed)


def word_cnn_model(x):
    # Input is BATCH_SIZE x MAX_DOCUMENT_LENGTH.
    # Word vector is BATCH_SIZE x MAX_DOCUMENT_LENGTH x EMBEDDING SIZE
    # Reshape to BATCH_SIZE x 100 x 2 x 1
    #            BATCH_SIZE x HEIGHT x WIDTH x CHANNELS

  word_vectors = tf.reshape(
      tf.contrib.layers.embed_sequence( x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE), [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE, 1]
      )

  word_list = tf.unstack(word_vectors, axis=1)

  with tf.variable_scope('CNN_Layer1'):
    conv1 = tf.layers.conv2d(
        word_vectors,
        filters     = N_FILTERS,
        kernel_size = FILTER_SHAPE1,
        padding     = 'VALID',
        activation  = tf.nn.relu
        )
        # Output BATCH_SIZE x 100-20+1 x 20-20+1 x 10
        #        BATCH_SIZE x 81 x 1 x 10
    
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size   = POOLING_WINDOW,
        strides     = POOLING_STRIDE,
        padding     = 'SAME'
        )
        # Output BATCH_SIZE x 41 x 1 x 10

  with tf.variable_scope('CNN_Layer2'):
    conv2 = tf.layers.conv2d(
        pool1,
        filters     = N_FILTERS,
        kernel_size = FILTER_SHAPE2,
        padding     = 'VALID',
        activation  = tf.nn.relu
        )
        # Output BATCH_SIZE x 41-20+1 x 1-1+1 x 10
        #        BATCH_SIZE x 22 x 1 x 10
    
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size   = POOLING_WINDOW,
        strides     = POOLING_STRIDE,
        padding     = 'SAME'
        )
        # Output BATCH_SIZE x 11 x 1 x 10

      # Reduce_max take the maximumvalue form each Onehot vector (find the char), and squeze the [1] dimension
    pool2 = tf.squeeze( tf.reduce_max( pool2, 1 ), squeeze_dims = [1] )
      # Output BATCH_SIZE x 10

    # Create the onput, and do not use a sofmax function, because thie will be done later  
  logits = tf.layers.dense(
    pool2,
    MAX_LABEL,
    activation = None #tf.nn.softmax
    )

  return logits, word_vectors


def read_data_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

  
def main():
    global n_words
  
    x_train, y_train, x_test, y_test, n_words = read_data_words()



    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)


    logits, word_vector = word_cnn_model(x)
    y = tf.nn.softmax(logits)
    pred = tf.argmax(y,1)
  
##  print(logits.shape)
  
    # Optimizer
        # Do the the softmax cross enxtropy.
    entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
      labels = tf.one_hot( y_, MAX_LABEL ), logits=logits
      )
    )
    train_op = tf.train.AdamOptimizer(LR).minimize(entropy)

    correct_predictions = tf.equal( tf.argmax(tf.nn.softmax(logits),1), tf.argmax(tf.one_hot( y_, MAX_LABEL ),1) )
    correct_predictions = tf.cast(correct_predictions, tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)

    no_epochs = EPOCHS

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        # training
        loss = []
        tr_err, te_acc = [], [] # Test error and training accuracy

        N = len(x_train)
        idx = np.arange(N)
        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]
          

            #Do the batch learning
            for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
                sess.run([train_op, entropy, word_vector], {x: x_train[start:end], y_: y_train[start:end]})

            # Find the error and accuracy
            tr_err.append( entropy.eval(feed_dict = {x: x_train, y_: y_train}) )  
            te_acc.append( accuracy.eval(feed_dict = {x: x_test, y_: y_test}) )


            if e%1 == 0:
                print('iter: %d, entropy: %g, accuracy: %g'%(e, tr_err[e], te_acc[e]))


  ## Print

    plt.figure(1)
    plt.plot(range(no_epochs), tr_err)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Training error vs Iterations')
    plt.savefig('Figures_B2/trainErr.png')


    plt.figure(2)
    plt.plot(range(no_epochs), te_acc)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.savefig('Figures_B2/test_acc.png')
  

if __name__ == '__main__':
  print("main")
  main()

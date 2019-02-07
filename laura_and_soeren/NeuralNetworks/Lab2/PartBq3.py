import numpy as np
import pylab as plt
import pandas
import tensorflow as tf
import csv

import os
if not os.path.isdir('Figures_B3'):
    print('creating the figures folder')
    os.makedirs('Figures_B3')


MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 50
CHARS = 256

BATCH_SIZE = 128
EPOCHS = 100
LR = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 100
tf.set_random_seed(seed)
np.random.seed(seed)

def rnn_model(x):
        # Input is BATCH_SIZE x 100.   Onehot make it a BATCH_SIZE x 100 x 256 Matrix.
        # Reshape to BATCH_SIZE x 100 x 256
        #            BATCH_SIZE x HEIGHT x WIDTH
        
  input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256])

  word_list = tf.unstack(input_layer, axis=1)
  

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits

def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = read_data_chars()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    logits= rnn_model(x)
    
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
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
                sess.run([train_op, entropy], {x: x_train[start:end], y_: y_train[start:end]})

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
    plt.savefig('Figures_B3/trainErr.png')


    plt.figure(2)
    plt.plot(range(no_epochs), te_acc)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.savefig('Figures_B3/test_acc.png')
  
  
if __name__ == '__main__':
  main()

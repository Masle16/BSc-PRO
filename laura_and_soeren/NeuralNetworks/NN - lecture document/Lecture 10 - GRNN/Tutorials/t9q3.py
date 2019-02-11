#
# Tutorial 9, Question 3
#

import nltk
import numpy as np
import tensorflow as tf
import pylab

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

MAX_LENGTH = 40
HIDDEN_SIZE = 10
MAX_LABEL = 2

no_epochs = 500
lr = 0.001

seed = 10
tf.set_random_seed(seed)

data = ['I did not like the movie',
        'The movie was not good',
        'I watched the movie with great interest',
        'I have seen better movies',
        'Good to see that movie',
        'I am not a fan of movies',
        'I liked the movie great',
        'The movie was of interest to me',
        'I thought they could show interesting scenes',
        'The movie did not have good scenes',
        'Family did not like the movie at all']
    
targets = [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]

test_data = ['The movie was not interesting to me',
        'I liked the movie with great interest']


def vocabulary(strings):
    chars = sorted(list(set(list(''.join(strings).lower()))))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    vocab_size = len(chars)

    return vocab_size, char_to_ix


def preprocess(strings, char_to_ix):
    data_chars = [list(d.lower()) for _, d in enumerate(strings)]
    for i, d in enumerate(data_chars):
        if len(d)>MAX_LENGTH:
            d = d[:MAX_LENGTH]
        elif len(d) < MAX_LENGTH:
            d += [' '] * (MAX_LENGTH - len(d))
            
    data_ids = np.zeros([len(data_chars), MAX_LENGTH], dtype=np.int64)
    for i in range(len(data_chars)):
        for j in range(MAX_LENGTH):
            data_ids[i, j] = char_to_ix[data_chars[i][j]]
    return np.array(data_ids)


def char_rnn_model(x, vocab_size, keep_prob):

    byte_vectors = tf.one_hot(x, vocab_size)
    byte_list = tf.unstack(byte_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

    encoding = tf.nn.dropout(encoding, keep_prob)
  
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits


def main():

    vocab_size, char_to_ix = vocabulary(data)
    x_train = preprocess(data, char_to_ix)
    y_train = np.array(targets)
    x_test = preprocess(test_data, char_to_ix)


    
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_LENGTH])
    y_ = tf.placeholder(tf.int64)
    keep_prob = tf.placeholder(tf.float32)

    logits = char_rnn_model(x, vocab_size, keep_prob)
    probs = tf.nn.softmax(logits)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # training
    loss, acc = [], []
    for e in range(no_epochs):
        _, loss_, acc_  = sess.run([train_op, entropy, accuracy], {x: x_train, y_: y_train, keep_prob:0.7})
        loss.append(loss_), acc.append(acc_)

        if e%10 == 0:
          print('epoch: %d, entropy: %g, acc: %g'%(e, loss[e], acc[e]))

    probs_  = sess.run(probs, {x: x_test, keep_prob:1.0})

    print(probs_)
    
    pylab.figure()
    pylab.plot(range(len(loss)), loss)
    pylab.xlabel('epochs')
    pylab.ylabel('entropy')
    pylab.savefig('figures/t9q3_1.png')

    pylab.figure()
    pylab.plot(range(len(acc)), acc)
    pylab.xlabel('epochs')
    pylab.ylabel('accuracy')
    pylab.savefig('figures/t9q3_2.png')

    pylab.show()
  

if __name__ == '__main__':
    main()

#
# Tutorial 9, Question 2
#

import numpy as np
import tensorflow as tf
import pylab
import csv

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

seed = 100
tf.set_random_seed(seed)

no_layers = 1
lr = 0.001
max_patience = 3

class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cells_ = []
        for layer in range(no_layers):
            cells_.append(tf.nn.rnn_cell.LSTMCell(self.hidden_dim))
        cells = tf.nn.rnn_cell.MultiRNNCell(cells_)
        
        outputs, states = tf.nn.dynamic_rnn(cells, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())

            patience = max_patience
            min_test_err = float('inf')
            step = 0
            err = []
            while patience > 0:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
 #                   print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
                    print('step:%d, train err:%g test err:%g'%(step, train_err, test_err))
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                err.append(train_err)
                step += 1
            save_path = self.saver.save(sess, './figures/model.ckpt')
            print('Model saved to {}'.format(save_path))
            return err

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './figures/model.ckpt')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output

def load_series(filename, series_idx=1):
    try:
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile)
            data = [float(row[series_idx]) for row in csvreader if len(row) > 0]
            normalized_data = (data - np.mean(data)) / np.std(data)
        return normalized_data
    except IOError:
        return None

def split_data(data, percent_train=0.80):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)
    return train_data, test_data

def plot_results(train_x, predictions, actual, filename):
    pylab.figure()
    num_train = len(train_x)
    pylab.plot(list(range(num_train)), train_x, color='b', label='training data')
    pylab.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    pylab.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    pylab.legend()
    if filename is not None:
        pylab.savefig(filename)
    else:
        pylab.show()


if __name__ == '__main__':
    seq_size = 8
    predictor = SeriesPredictor(input_dim=1, seq_size=seq_size, hidden_dim=10)
    data = load_series('international-airline-passengers.csv')
    train_data, actual_vals = split_data(data)

    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])

    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i+seq_size], axis=1).tolist())
        test_y.append(actual_vals[i+1:i+seq_size+1])

    print(np.array(test_x).shape)

    err = predictor.train(train_x, train_y, test_x, test_y)

    pylab.figure()
    pylab.plot(range(len(err)), err)
    pylab.xlabel('epochs')
    pylab.ylabel('mean square error')
    pylab.savefig('./figures/t9q2_1.png')

    with tf.Session() as sess:
        predicted_vals = predictor.test(sess, test_x)[:,0]
        print('predicted_vals', np.shape(predicted_vals))
        plot_results(train_data, predicted_vals, actual_vals, './figures/t9q2_2.png')

        prev_seq = train_x[-1]
        predicted_vals = []
        for i in range(len(data)-len(train_x)):
            next_seq = predictor.test(sess, [prev_seq])
            predicted_vals.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        plot_results(train_data, predicted_vals, actual_vals, './figures/t9q2_3.png')

    pylab.show()    

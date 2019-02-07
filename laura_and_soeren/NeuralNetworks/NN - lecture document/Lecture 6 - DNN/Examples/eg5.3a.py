#
#   Chapter 5, example 3a
#

import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow as tf
import pylab as plt

learning_rate = 0.001

no_features = 13
no_labels = 1
no_iters = 2500

seed = 10
tf.set_random_seed(seed)

def ffn(x, hidden_units):
  
  # Hidden
  with tf.name_scope('hidden'):
    weights = tf.Variable(
      tf.truncated_normal([no_features, hidden_units],
                            stddev=1.0 / np.sqrt(float(no_features))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),
                         name='biases')
    hidden = tf.nn.relu(tf.matmul(x, weights) + biases)
    
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units, no_labels],
                            stddev=1.0 / np.sqrt(float(hidden_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([no_labels]),
                         name='biases')
    logits = tf.matmul(hidden, weights) + biases
    
  return logits


def main():

  # Load dataset
  boston = datasets.load_boston()
  x, y = boston.data, boston.target

  # Split dataset into train / test
  x_train, x_test, y_train_, y_test_ = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42)
  y_train = y_train_.reshape(len(y_train_), no_labels)
  y_test = y_test_.reshape(len(y_test_), no_labels)

  # Scale data (training set) to 0 mean and unit standard deviation.
  scaler = preprocessing.StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.fit_transform(x_test)

  x = tf.placeholder(tf.float32, [None, no_features])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, no_labels])

  # Build the graph for the deep net
  y = ffn(x, 10)

  error = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), axis = 1))

  train = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      tr_err, te_err = [], []
      for i in range(no_iters):
         train.run(feed_dict={x: x_train, y_: y_train})

         tr_err.append(error.eval(feed_dict={x: x_train, y_: y_train}))
         te_err.append(error.eval(feed_dict={x: x_test, y_: y_test}))
         if i % 100 == 0:
            print('step %d, error: train %g, test %g' % (i, tr_err[i], te_err[i]))
            
  # plot learning curves
  plt.figure(1)
  plt.plot(range(no_iters), tr_err, label = 'train error')
  plt.plot(range(no_iters), te_err, label = 'test error')
  plt.xlabel('iterations')
  plt.ylabel('mean square error')
  plt.title('GD learning')
  plt.legend()
  plt.savefig('figures/5.3a_1.png')


  plt.show()

    

if __name__ == '__main__':
  main()


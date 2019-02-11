#
#   Chapter 5, example 3b
#

import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow as tf
import pylab as plt
import multiprocessing as mp

learning_rate = 0.001

no_features = 13
no_labels = 1
no_iters = 2500

seed = 10
tf.set_random_seed(seed)


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


def my_train(hidden_units):

  x = tf.placeholder(tf.float32, [None, no_features])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, no_labels])

  # Build the graph for the deep net
  y = ffn(x, hidden_units)

  error = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), axis = 1))

  train = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    te_err = []
    for i in range(no_iters):
      train.run(feed_dict={x: x_train, y_: y_train})

      te_err.append(error.eval(feed_dict={x: x_test, y_: y_test}))

      if i % 100 == 0:
        print('%d: step %d, test error %g' % (hidden_units, i, te_err[i]))

  return te_err
               


def main():

  hidden_units = [1, 4, 16, 64]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  cost = p.map(my_train, hidden_units)

    # plot learning curves
  plt.figure(1)

  min_cost = []
  for h in range(len(hidden_units)):
    plt.plot(range(no_iters), cost[h], label = 'hidden = {}'.format(hidden_units[h]))
    min_cost.append(min(cost[h]))


  plt.xlabel('iterations')
  plt.ylabel('mean square error')
  plt.title('GD learning')
  plt.legend()
  plt.savefig('figures/5.3b_1.png')

  
  plt.figure(2)
  plt.plot(hidden_units, min_cost)
  plt.xlabel('number of hidden neurons')
  plt.ylabel('test error')
  plt.title('test error vs. number of hidden neurons')
  plt.savefig('figures/5.3b_2.png')

  plt.show()

    

if __name__ == '__main__':
  main()


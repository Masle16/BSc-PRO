#
# Chapter 4, Example 4b
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import pylab as plt

import tensorflow as tf

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
seed = 100

tf.logging.set_verbosity(tf.logging.ERROR)

# Load dataset
boston = datasets.load_boston()
x, y = boston.data, boston.target

# Split dataset into train / test
x_train, x_test, y_train, y_test = model_selection.train_test_split(
x, y, test_size=0.2, random_state=42)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_transformed = scaler.transform(x_test)
  

def train(network):

  tf.set_random_seed(seed)

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  feature_columns = [
      tf.feature_column.numeric_column('x', shape=np.array(x_train).shape[1:])]
  regressor = tf.estimator.DNNRegressor(
      feature_columns=feature_columns, hidden_units= network)

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_train}, y=y_train, batch_size=32, num_epochs=None, shuffle=True)
  regressor.train(input_fn=train_input_fn, steps=2000)

  tf.reset_default_graph()
  # Predict.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_transformed}, y=y_test, num_epochs=1, shuffle=False)

  # Score with tensorflow.
  scores = regressor.evaluate(input_fn=test_input_fn)
  print('MSE (tensorflow): {0:f}'.format(scores['average_loss']))

  return scores['average_loss']


def main():

  networks = [[5], [5, 5], [5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5, 5]]

  error = []
  for network in networks:
    error.append(train(network))
  
  plt.figure()
  plt.plot(np.arange(len(networks)), error)
  plt.xlabel('number of hidden layers')
  plt.ylabel('MSE')
  plt.xticks(np.arange(len(networks)), [1, 2, 3, 4, 5])
  plt.title('error vs. number of hidden layers')
  plt.savefig('./figures/5.4b_1.png')

  plt.show()


if __name__ == '__main__':
  main()

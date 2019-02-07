#
# Tutorial 4, Question 2
#

from sklearn import datasets
import numpy as np
import tensorflow as tf
import pylab as plt
import multiprocessing as mp
import time

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')



no_epochs = 1000
lr = 0.01

seed = 10
np.random.seed(seed)

# input data
iris = datasets.load_iris()
iris.data -= np.mean(iris.data, axis=0)

X = iris.data
Y = np.zeros((len(X), 3))
for i in range(len(X)):
    Y[i, iris.target[i]] = 1

idd = np.arange(len(X))
np.random.shuffle(idd)
X, Y = X[idd], Y[idd]

trainX, trainY = X[:120], Y[:120]
testX, testY = X[120:], Y[120:]

no_data = len(trainX)


def train_batch(batch_size):

    tf.set_random_seed(seed)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 4])
    W = tf.Variable(tf.truncated_normal([4, 3],stddev=1.0 / np.sqrt(4)))
    b = tf.Variable(tf.zeros([3]))
    u = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3])

    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u))
    error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(u, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))
    train = tf.train.GradientDescentOptimizer(lr).minimize(entropy)

    # train

    idx = np.arange(no_data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        time_to_epoch = 0
        for i in range(no_epochs):

            np.random.shuffle(idx)
            train_X, train_Y = trainX[idx], trainY[idx]

            t = time.time()
            for start, end in zip(range(0, no_data, batch_size), range(batch_size, no_data, batch_size)):
                train.run(feed_dict={x: train_X[start:end], y_: train_Y[start:end]})

            time_to_epoch += time.time() - t

            entropy_ = entropy.eval(feed_dict={x: train_X, y_: train_Y})           
            error_ = error.eval(feed_dict={x: testX, y_: testY})
    
            if i%10 == 0:
                print('batch %d, epoch %d, error %d entropy %g'%(batch_size, i, error_, entropy_))

    return np.array([error_, entropy_, time_to_epoch*1000/(no_epochs)])



def main():

#  batch_sizes = [16]
  batch_sizes = [2, 4, 8, 16, 24, 32, 48, 64]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  paras = p.map(train_batch, batch_sizes)

  paras = np.array(paras)

  error, entropy, time_to_epoch = paras[:,0], paras[:,1], paras[:,2]


  plt.figure(1)
  plt.plot(range(len(batch_sizes)), error)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('test error')
  plt.title('test error vs. batch size')
  plt.savefig('./figures/t4q2b_1.png')

  plt.figure(2)
  plt.plot(range(len(batch_sizes)), entropy)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('entropy')
  plt.title('entropy vs. batch size')
  plt.savefig('./figures/t4q2b_2.png')

  plt.figure(3)
  plt.plot(range(len(batch_sizes)), time_to_epoch)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('time to epoch (ms)')
  plt.title('time to epoch vs. batch size')
  plt.savefig('./figures/t4q2b_3.png')
 
  plt.show()
  

if __name__ == '__main__':
  main()



        

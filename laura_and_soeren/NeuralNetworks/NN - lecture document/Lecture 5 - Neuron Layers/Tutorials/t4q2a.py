#
# Tutorial 4, Question 2
#

from sklearn import datasets
import numpy as np
import tensorflow as tf
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')



no_epochs = 1000
batch_size = 16
lr = 0.01

seed = 10
np.random.seed = 10
tf.set_random_seed(seed)

# input data
iris = datasets.load_iris()
iris.data -= np.mean(iris.data, axis=0)

print(iris.target.shape)

X = iris.data
Y = np.zeros((len(X), 3))
for i in range(len(X)):
    Y[i, iris.target[i]] = 1

idx = np.arange(len(X))
np.random.shuffle(idx)
X, Y = X[idx], Y[idx]

trainX, trainY = X[:120], Y[:120]
testX, testY = X[120:], Y[120:]

no_data = len(trainX)

# Create the model
x = tf.placeholder(tf.float32, [None, 4])
W = tf.Variable(tf.truncated_normal([4, 3],stddev=1.0 / np.sqrt(4)))
b = tf.Variable(tf.zeros([3]))
u = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 3])

entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u))

error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(u, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))
  
train = tf.train.GradientDescentOptimizer(lr).minimize(entropy)

# train
tf.InteractiveSession()
tf.global_variables_initializer().run()


error_, entropy_ = [], []
idx = np.arange(no_data)

for i in range(no_epochs):

    np.random.shuffle(idx)
    train_X, train_Y = trainX[idx], trainY[idx]

    for start, end in zip(range(0, no_data, batch_size), range(batch_size, no_data, batch_size)):
        train.run(feed_dict={x: train_X[start:end], y_: train_Y[start:end]})
      

    error_.append(error.eval(feed_dict={x: testX, y_: testY}))
    entropy_.append(entropy.eval(feed_dict={x: train_X, y_: train_Y}))
    
    if i%10 == 0:
        print('epoch %d, entropy %g, errors %d'%(i, entropy_[i], error_[i]))


plt.figure(1)
plt.plot(range(no_epochs), entropy_)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/t4q2a_1.png')

plt.figure(2)
plt.plot(range(no_epochs), error_)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/t4q2a_2.png')


plt.show()


        

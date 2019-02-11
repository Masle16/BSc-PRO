#
# Chapter 4, example 4
#


from sklearn import datasets
import numpy as np
import tensorflow as tf
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


no_iters = 1000

# input data
iris = datasets.load_iris()
iris.data -= np.mean(iris.data, axis=0)
n = iris.data.shape[0]

print(iris.target.shape)

X = iris.data
no_data = len(iris.data)
Y = np.zeros((no_data, 3))
for i in range(no_data):
    Y[i, iris.target[i]] = 1


# Create the model
x = tf.placeholder(tf.float32, [None, 4])
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))
u = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=u))

error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(u, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))
  
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# train
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

error_, entropy_ = [], []
for i in range(no_iters):
    sess.run(train, feed_dict={x: X, y_: Y})
    entropy_.append(sess.run(cross_entropy, feed_dict={x:X, y_:Y}))
    error_.append(sess.run(error, feed_dict={x:X, y_:Y}))
    if i%100 == 0:
        print('iter:{}, error:{}'.format(i, error_[i]))


plt.figure(1)
plt.plot(range(no_iters), entropy_)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/4.4_1.png')

plt.figure(2)
plt.plot(range(no_iters), error_)
plt.xlabel('epochs')
plt.ylabel('classification error')
plt.savefig('./figures/4.4_2.png')


plt.show()


        

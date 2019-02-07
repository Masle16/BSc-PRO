#
# Chapter 10, Example 1
#


import numpy as np
import tensorflow as tf
import pylab as plt


import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
no_epochs = 2000
lr = 0.1

n_in = 5
n_hidden = 3

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

x = tf.placeholder(tf.float32, [None, n_in])

W = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=1/np.sqrt(n_in)))
b = tf.Variable(tf.zeros([n_hidden]))
b_prime = tf.Variable(tf.zeros([n_in]))
W_prime = tf.transpose(W)

h = tf.sigmoid(tf.matmul(x, W) + b)
y = tf.sigmoid(tf.matmul(h, W_prime) + b_prime)
o = tf.where(tf.greater(y, 0.5), tf.ones(tf.shape(y)), tf.zeros(tf.shape(y)))

cost = - tf.reduce_mean(tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), axis=1))
bit_cost = tf.reduce_sum(tf.cast(tf.not_equal(x,o), tf.int32))

train = tf.train.GradientDescentOptimizer(lr).minimize(cost)

X = np.array([[1, 0, 1, 0, 0],
              [0, 0, 1, 1, 0],
              [1, 1, 0, 1, 1],
              [0, 1, 1, 1, 0]]).astype(np.float32)

print(X)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

c, bc = [], [] 
for i in range(no_epochs):
    train.run(feed_dict={x: X})
    c_, bc_ = sess.run([cost, bit_cost], {x: X})
    c.append(c_)
    bc.append(bc_)
    if i%100 == 0:
        print('epoch: %d, b_cost: %d, cost: %g'%(i, bc[i], c[i]))
    

W_, b_, b_prime_ = sess.run([W, b, b_prime])
print('W: {}'.format(W_))
print('b: {}'.format(b_))
print('b_prime: {}'.format(b_prime_))

h_, y_, o_ = sess.run([h, y, o],{x: X})

print(h_)
print(y_)
print(o_)

plt.figure()
plt.plot(range(no_epochs), c)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/10.1_1.png')

plt.figure()
plt.plot(range(no_epochs), bc)
plt.xlabel('epochs')
plt.ylabel('bit_errors')
plt.savefig('./figures/10.1_2.png')

plt.show()

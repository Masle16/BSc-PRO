#
# Tutorial 10, Question 1a
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

n_in = 9
n_hidden = 4

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

entropy = -tf.reduce_mean(tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), axis=1))
bit_cost = tf.reduce_sum(tf.cast(tf.not_equal(x,o), tf.int32))

train = tf.train.GradientDescentOptimizer(lr).minimize(entropy)

X = np.array([[[1, 1, 1],[0, 0, 0], [0, 0, 0]],
              [[1, 0, 0],[1, 0, 0], [1, 0, 0]],
              [[0, 0, 1],[0, 1, 0], [1, 0, 0]],
              [[1, 0, 0],[0, 1, 0], [0, 0, 1]],
              [[1, 0, 0], [1, 0, 0], [1, 1, 1]]]).astype(np.float32)

X = X.reshape(5, 9)
print(X)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

c, bc = [], [] 
for i in range(no_epochs):
    train.run(feed_dict={x: X})
    c_, bc_ = sess.run([entropy, bit_cost], {x: X})
    c.append(c_)
    bc.append(bc_)
    if i%100 == 0:
        print('epoch: %d, b_cost: %d, cost: %g'%(i, bc[i], c[i]))
    

W_, b_, b_prime_ = sess.run([W, b, b_prime])
print('W: {}'.format(W_.reshape(3,3, 4).transpose([2, 0, 1])))
print('b: {}'.format(b_))
print('b_prime: {}'.format(b_prime_))

h_, y_, o_ = sess.run([h, y, o],{x: X})

print('h: {}'.format(h_))
print('y: {}'.format(y_.reshape(5, 3,3)))
print('o: {}'.format(o_.reshape(5, 3, 3)))

plt.figure()
plt.plot(range(no_epochs), c)
plt.xlabel('epochs')
plt.ylabel('cross-entropy')
plt.savefig('./figures/t10q1a_1.png')

plt.figure()
plt.plot(range(no_epochs), bc)
plt.xlabel('epochs')
plt.ylabel('bit_errors')
plt.savefig('./figures/t10q1a_2.png')

plt.figure()
plt.gray()
plt.subplot(1, 4, 1); plt.axis('off'); plt.imshow(W_[:,0].reshape(3,3))
plt.subplot(1, 4, 2); plt.axis('off'); plt.imshow(W_[:,1].reshape(3,3))
plt.subplot(1, 4, 3); plt.axis('off'); plt.imshow(W_[:,2].reshape(3,3))
plt.subplot(1, 4, 4); plt.axis('off'); plt.imshow(W_[:,3].reshape(3,3))
plt.savefig('./figures/t10q1a_3.png')

plt.show()

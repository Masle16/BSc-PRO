#
# Tutorial 5, Question 3b
#

import tensorflow as tf
import numpy as np
import pylab as plt
import multiprocessing as mp

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    

num_iters = 10000

num_features = 2
num_hidden1 = 4
num_hidden2 = 3

seed = 100
np.random.seed(seed)
tf.set_random_seed(seed)


# generate data
X = np.zeros((9*9, num_features))
p = 0
for i in np.arange(-1, 1.001, 0.25):
    for j in np.arange(-1, 1.001, 0.25):
        X[p] = [i, j]
        p += 1
np.random.shuffle(X)
Y = np.zeros((9*9, 1))
Y[:,0] = 0.8*X[:,0]**2 - X[:,1]**3 + 2.5*X[:,0]*X[:,1]

#Define variables:
W1 = tf.Variable(tf.truncated_normal([num_features, num_hidden1],
                            stddev=1.0 / np.sqrt(float(num_features)) , dtype=tf.float32))
b1 = tf.Variable(tf.zeros([num_hidden1]), dtype=tf.float32)
W2 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2],
                            stddev=1.0 / np.sqrt(float(num_hidden1)) , dtype=tf.float32))
b2 = tf.Variable(tf.zeros([num_hidden2]), dtype=tf.float32)
W3 = tf.Variable(tf.truncated_normal([num_hidden2, 1],
                            stddev=1.0 / np.sqrt(float(num_hidden2)) , dtype=tf.float32))
b3 = tf.Variable(tf.zeros([1]), dtype=tf.float32)


# Model input and output
x = tf.placeholder(tf.float32, [None, X.shape[1]])
d = tf.placeholder(tf.float32, [None, Y.shape[1]])
lr = tf.placeholder(tf.float32)

u1 = tf.matmul(x, W1) + b1
h1 = tf.nn.relu(u1)
u2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.relu(u2)
y = tf.matmul(h2, W3) + b3

cost = tf.reduce_mean(tf.square(d - y))

grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = tf.gradients(cost, [W1, b1, W2, b2, W3, b3])


W3_new, b3_new = W3.assign(W3 - lr*grad_W3), b3.assign(b3 - lr*grad_b3)
W2_new, b2_new = W2.assign(W2 - lr*grad_W2), b2.assign(b2 - lr*grad_b2)
W1_new, b1_new = W1.assign(W1 - lr*grad_W1), b1.assign(b1 - lr*grad_b1)


def my_train(rate):
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong

    err = []
    for i in range(num_iters):
      sess.run([W1_new, b1_new, W2_new, b2_new, W3_new, b3_new], {x:X, d:Y, lr:rate})
      err.append(sess.run(cost, {x:X, d:Y}))

      if not i%100:
          print('rate:{}, epoch:{}, mse:{}'.format(rate, i,err[i]))
                    
    return(err)


def main():

    rates = [0.005, 0.01, 0.05, 0.1]
    
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    costs = p.map(my_train, rates)

    plt.figure()
    for r in range(len(rates)):
      plt.plot(range(num_iters), costs[r], label='lr = {}'.format(rates[r]))

    plt.xlabel('iterations')
    plt.ylabel('mean square error')
    plt.title('gradient descent learning')
    plt.legend()
    plt.savefig('./figures/t5q3b_1.png')

    plt.show()


if __name__ == '__main__':
  main()


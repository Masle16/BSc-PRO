import tensorflow as tf
import numpy as np
import pylab as plt
import multiprocessing as mp
from functools import partial

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')
	
no_data = 16
no_iters = 500
SEED = 10
np.random.seed(SEED)

## training data
X = np.random.rand(no_data,2)
Y = 1.0 +3.3*X[:,0]**2-2.5*X[:,1]+0.2*X[:,0]*X[:,1]
Y = Y.reshape(no_data,1)

# Model parameters
w = tf.Variable(0.01*np.random.rand(2), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)
lr = tf.Variable(0.0001, dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
d = tf.placeholder(tf.float32)

u = tf.tensordot(x, w, axes=1) + b
y = 6.0*tf.sigmoid(u) - 1.5
loss = tf.reduce_sum(tf.square(d - y)) # sum of the squares
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)


def my_train(alpha):
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init) # reset values to wrong
  sess.run(lr.assign(alpha))

  cost = []
  idx = np.arange(no_data)
  XX, YY = X, Y
  for i in range(no_iters):
    np.random.shuffle(idx)
    XX, YY = XX[idx], YY[idx]
    cost_ = []
    for p in range(len(XX)):
      sess.run(train, {x: XX[p], d: YY[p]})
      cost_.append(sess.run(loss, {x: XX[p], d: YY[p]}))
    cost.append(np.sum(cost_)/no_data)

    if not i%100:
      print(cost[i])

  return cost
  

rates = [0.005, 0.01, 0.05, 0.1]


no_threads = mp.cpu_count()
p = mp.Pool(processes = no_threads)
costs = p.map(my_train, rates)

plt.figure()
for r in range(len(rates)):
  plt.plot(range(no_iters), costs[r], label='lr = {}'.format(rates[r]))
plt.xlabel('iterations')
plt.ylabel('cost')
plt.title('stochastic gradient descent')
plt.legend()
plt.savefig('./figures/2.4a_1.png')

plt.show()



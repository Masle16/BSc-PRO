import tensorflow as tf
import numpy as np
import pylab as plt
import multiprocessing as mp

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')
	
no_data = 16
no_iters = 500
SEED = 10
np.random.seed(SEED)

# training data
X = np.random.rand(no_data,2)
Y = 1.0 +3.3*X[:,0]**2-2.5*X[:,1]+0.2*X[:,0]*X[:,1]
Y = Y.reshape(no_data,1)

# Model parameters
w = tf.Variable(0.01*np.random.rand(2,1), dtype=tf.float32)
b = tf.Variable(0., dtype=tf.float32)
lr = tf.Variable(0.01, dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, [X.shape[0], 2])
d = tf.placeholder(tf.float32, [Y.shape[0], 1])

u = tf.matmul(x,w) + b
y = 6.0*tf.sigmoid(u) - 1.5
loss = tf.reduce_sum(tf.square(d - y)) # sum of the squares
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)


def my_train(rate):
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init) # reset values to wrong
  sess.run(lr.assign(rate))
 
  print(rate)

  err = []
  for i in range(no_iters):
    sess.run(train, {x: X, d: Y})
    loss_ = sess.run(loss/no_data, {x: X, d: Y})
    err.append(loss_)

  return err
  

no_threads = mp.cpu_count()
rates = [0.005, 0.01, 0.05, 0.1]

p = mp.Pool(processes = no_threads)
results = p.map(my_train, rates)

plt.figure()
for i in range(len(rates)):
  plt.plot(range(no_iters), results[i], label='lr = {}'.format(rates[i]))
plt.xlabel('epochs')
plt.ylabel('cost')
plt.title('gradient descent')
plt.legend()
plt.savefig('./figures/2.4b_1.png')

plt.show()



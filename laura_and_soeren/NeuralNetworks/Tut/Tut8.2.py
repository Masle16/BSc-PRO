# input 28x28, feature maps x2,
# 25  Filters of 9x9
# Polling window of 4x4 

import tensorflow as tf
import numpy as np
import pylab 

## Define parameters
n_input = 3
n_hidden = 5
n_output = 2
n_steps = 100
n_seqs = 8

LR = 10**(-2) # And adam
EPOCHS = 10000

SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)


x_train = np.random.rand(n_seqs, n_steps, n_input)
y_train = np.zeros([n_seqs, n_steps, n_output])

y_train[:,1:,0] = 5*x_train[:,1:,0] - x_train[:,:-1,2]
y_train[:,3:,1] = 25*x_train[:,2:-1, 1]*x_train[:,:-3,2]

y_train += 0.1*np.random.randn(n_seqs, n_steps, n_output)

# Build the graph
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_steps, n_output])
initial_state = tf.placeholder(tf.float32, [None, n_hidden])

U = tf.Variable(tf.truncated_normal( (n_input, n_hidden), stddev=1.0 / np.sqrt(n_input))) 
W = tf.Variable(tf.truncated_normal( (n_hidden, n_hidden), stddev=1.0 / np.sqrt(n_hidden)))
b = tf.Variable(np.zeros([n_hidden]), dtype= tf.float32)
V = tf.Variable(tf.truncated_normal( (n_hidden, n_output), stddev=1.0 / np.sqrt(n_hidden)))
c = tf.Variable(np.zeros([n_output]), dtype= tf.float32)




    # Go through all the time steps
h = initial_state
ys = []
for i,x_ in enumerate(tf.split(x, n_steps, axis =1)):
    h = tf.tanh(tf.matmul(tf.squeeze(x_), U) + tf.matmul(h,W) + b)
    y_ = tf.matmul(h, V) + c
    ys.append(y_)

ys_ = tf.stack(ys, axis=1)
cost = tf.reduce_mean( tf.reduce_sum( tf.square(y - ys_), axis=2))
train_op = tf.train.AdamOptimizer(LR).minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

state = np.zeros([n_seqs, n_hidden])
loss = []

for i in range(EPOCHS):
    sess.run(train_op, {x:x_train, y:y_train, initial_state:state})
    loss.append(sess.run(cost,  {x:x_train, y:y_train, initial_state:state}))

    if not i % 100:
        print('iter:%d, cost: %g'%(i, loss[i]))

pred = sess.run(ys_, {x:x_train, y: y_train, initial_state: state})

pylab.figure()
pylab.plot(range(EPOCHS), loss)
pylab.xlabel('epochs')
pylab.ylabel('mean square error')
#pylab.savefig('./figures/t8q2_1.png')

pylab.figure(figsize=(4, 2))
for i in range(n_seqs):
    ax = pylab.subplot(4, 2, i+1)
    pylab.axis('off')
    ax.plot(range(n_steps), y_train[i,:,0])
    #pylab.savefig('./figures/t8q2_2.png')

pylab.figure(figsize=(4, 2))
for i in range(n_seqs):
    ax = pylab.subplot(4, 2, i+1)
    pylab.axis('off')
    ax.plot(range(n_steps), y_train[i,:,1])
    #pylab.savefig('./figures/t8q2_3.png')
    

pylab.figure(figsize=(4, 2))
for i in range(n_seqs):
    ax = pylab.subplot(4, 2, i+1)
    pylab.axis('off')
    ax.plot(range(n_steps), y_train[i,:,0])
    ax.plot(range(n_steps), pred[i, :, 0])
    #pylab.savefig('./figures/t8q2_4.png')

pylab.figure(figsize=(4, 2))
for i in range(n_seqs):
    ax = pylab.subplot(4, 2, i+1)
    pylab.axis('off')
    ax.plot(range(n_steps), y_train[i,:,1])
    ax.plot(range(n_steps), pred[i, :, 1])
    #pylab.savefig('./figures/t8q2_5.png')

pylab.show()








        

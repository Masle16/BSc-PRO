#
# Chapter 11, Example 1
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

mu,sigma=0,1

lr = 0.001

no_epochs=500
batch_size = 128
n_samples = 10000


n_in = 1
n_hidden1 = 5
n_hidden2 = 5

n_in_gen = 1
n_hidden1_gen = 5
n_hidden2_gen = 3


seed = 100
np.random.seed(seed)
tf.set_random_seed(seed)

if not os.path.exists('out/'):
    os.makedirs('out/')

def sample_x(m):
    s = np.random.normal(mu, sigma, [m, n_in])
    s[s > 5] = mu
    s[s < -5] = mu
    return s

def sample_z(m):
    return np.random.uniform(-1, +1, [m, n_in_gen])

def plot(samples, logits):
    # plots pg, pdata, decision boundary 
    fig, ax = plt.subplots(1)
    # p_data
    xs = np.linspace(-5,5,n_samples)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')

    ax.plot(xs, logits, label='decision boundary')

    # distribution of inverse-mapped points

    histc, edges = np.histogram(samples, bins = 10)
    ax.plot(np.linspace(-5,5,10), histc/n_samples, label='p_g')

    # ylim, legend
    ax.set_ylim(0,1.1)
    plt.legend()

    return fig

G_W1 = tf.Variable(tf.truncated_normal([n_in_gen, n_hidden1_gen], stddev=1/np.sqrt(n_in_gen)))
G_b1 = tf.Variable(tf.zeros(shape=[n_hidden1_gen]))
G_W2 = tf.Variable(tf.truncated_normal([n_hidden1_gen, n_hidden2_gen], stddev=1/np.sqrt(n_hidden1_gen)))
G_b2 = tf.Variable(tf.zeros(shape=[n_hidden2_gen]))
G_W3 = tf.Variable(tf.truncated_normal([n_hidden2_gen, n_in], stddev=1/np.sqrt(n_hidden2_gen)))
G_b3 = tf.Variable(tf.zeros(shape=[n_in]))

D_W1 = tf.Variable(tf.truncated_normal([n_in, n_hidden1], stddev=1/np.sqrt(n_in)))
D_b1 = tf.Variable(tf.zeros(shape=[n_hidden1]))
D_W2 = tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=1/np.sqrt(n_hidden1)))
D_b2 = tf.Variable(tf.zeros(shape=[n_hidden2]))
D_W3 = tf.Variable(tf.truncated_normal([n_hidden2, 1], stddev=1/np.sqrt(n_hidden2)))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

def generator(z):
    
    G_h1 = tf.nn.sigmoid(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.sigmoid(tf.matmul(G_h1, G_W2) + G_b2)
    G_out = tf.nn.tanh(tf.matmul(G_h2, G_W3) + G_b3)

    return G_out 


def discriminator(x):

    D_h1 = tf.nn.sigmoid(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.sigmoid(tf.matmul(D_h1, D_W2) + D_b3)
    D_logits = tf.matmul(D_h2, D_W3) + D_b3
    D_probs = tf.nn.sigmoid(D_logits)

    return D_logits, D_probs


x = tf.placeholder(tf.float32, shape=[None, n_in])
z = tf.placeholder(tf.float32, shape=[None, n_in_gen])

G_sample = generator(z)
D_real = discriminator(x)
D_fake, D_fake_probs  = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=0.99*tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=0.99*tf.ones_like(D_fake)))

D_solver = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=[D_W1, D_W2, D_b1, D_b2])
G_solver = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=[G_W1, G_W2, G_b1, G_b2])




sess = tf.Session()
sess.run(tf.global_variables_initializer())




trainX = sample_x(n_samples)
i = 0
d_loss, g_loss = [], []
for e in range(no_epochs):
    if e % 10 == 0:
        samples, probs = sess.run([G_sample, D_fake_probs], feed_dict={z: sample_z(n_samples)})

        fig = plot(samples, probs)
        plt.savefig('out/1_{}.png'.format(str(e).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    d_cost, g_cost  = [], []
    for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
    
        _, D_loss_ = sess.run([D_solver, D_loss], feed_dict={x:trainX[start:end], z: sample_z(batch_size)})

        for k in range(2):
            _, G_loss_ = sess.run([G_solver, G_loss], feed_dict= {z: sample_z(batch_size)})
        d_cost.append(D_loss_), g_cost.append(G_loss_)
        
    d_loss.append(np.mean(d_cost))
    g_loss.append(np.mean(g_cost))
   

    if e %10 == 0:
        print('epoch: {}, D loss: {:.4}, G_loss: {:.4}'.format(e, D_loss_, G_loss_))

plt.figure()
plt.plot(range(no_epochs), d_loss, label='discriminator')
plt.plot(range(no_epochs), g_loss, label='generator')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('out/11.1_1.png')

plt.show()

        

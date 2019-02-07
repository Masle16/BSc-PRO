import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


n_in = 784
n_hidden = 128
n_in_gen = 100
n_hidden_gen = 256

no_epochs = 1000
batch_size = 128

seed = 100
np.random.seed(seed)
tf.set_random_seed(seed)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n]).astype(np.float32)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

G_W1 = tf.Variable(tf.truncated_normal([n_in_gen, n_hidden_gen], stddev=1/np.sqrt(n_in_gen)))
G_b1 = tf.Variable(tf.zeros(shape=[n_hidden_gen]))
G_W2 = tf.Variable(tf.truncated_normal([n_hidden_gen, n_in], stddev=1/np.sqrt(n_hidden_gen)))
G_b2 = tf.Variable(tf.zeros(shape=[n_in]))

D_W1 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=1/np.sqrt(n_in)))
D_b1 = tf.Variable(tf.zeros(shape=[n_hidden]))
D_W2 = tf.Variable(tf.truncated_normal([n_hidden, 1], stddev=1/np.sqrt(n_hidden)))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

def generator(z):
    
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_out = tf.nn.sigmoid(tf.matmul(G_h1, G_W2) + G_b2)

    return G_out 


def discriminator(x):

    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logits = tf.matmul(D_h1, D_W2) + D_b2

    return D_logits


x = tf.placeholder(tf.float32, shape=[None, n_in])
z = tf.placeholder(tf.float32, shape=[None, n_in_gen])

G_sample = generator(z)
D_real = discriminator(x)
D_fake  = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=0.99*tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=0.99*tf.ones_like(D_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=[D_W1, D_W2, D_b1, D_b2])
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=[G_W1, G_W2, G_b1, G_b2])



mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
trainX = mnist.train.images

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

idx = np.arange(len(trainX))
d_loss, g_loss = [], []
for e in range(no_epochs):
    np.random.shuffle(idx)
    trainX = trainX[idx]

    d_cost, g_cost  = [], []
    for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
    
        _, D_loss_ = sess.run([D_solver, D_loss], feed_dict={x:trainX[start:end], z: sample_Z(batch_size, n_in_gen)})
        _, G_loss_ = sess.run([G_solver, G_loss], feed_dict= {z: sample_Z(batch_size, n_in_gen)})
        d_cost.append(D_loss_), g_cost.append(G_loss_)
        
    d_loss.append(np.mean(d_cost))
    g_loss.append(np.mean(g_cost))

    samples = sess.run(G_sample, feed_dict={z: sample_Z(16, n_in_gen)})
    fig = plot(samples)
    plt.savefig('out/2_{}.png'.format(str(e).zfill(4)), bbox_inches='tight')
    plt.close(fig)

    print('epoch: {}, D loss: {:.4}, G_loss: {:.4}'.format(e, d_loss[e], g_loss[e]))


plt.figure()
plt.plot(range(no_epochs), d_loss, label='discriminator')
plt.plot(range(no_epochs), g_loss, label='generator')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('out/11.2_1.png')


plt.show()

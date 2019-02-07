## Design a perceptron layer to perform the following mapping with SGD

import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIVLE_DEVICES"]="-1"

if not os.path.isdir('figures'):
    print('Creating a figure folder')
    os.makedirs('figures')


w = tf.Variable( [[1 , -2], [2,0]], dtype=tf.float32 )
b = tf.Variable( [[3, .1]] , dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)

#z = x
z =  tf.matmul( tf.transpose(w), x ) 
#y = tf.sigmoid(z)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
y_   = sess.run( z, {x:[1.0, 3.0] } )
print('y: {}'.format(y_))



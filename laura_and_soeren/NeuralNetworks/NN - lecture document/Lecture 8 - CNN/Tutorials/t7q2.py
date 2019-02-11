import tensorflow as tf
import numpy as np
import pylab
from PIL import Image

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')
    
seed = 10
np.random.seed(10)

# open image and normalize
img = Image.open('3wolfmoon.jpg')
img = np.asarray(img, dtype='float32') / 256.
print(img.shape)
img_shape = [1, img.shape[0], img.shape[1], img.shape[2]]


# weights and biases
w_shp = (9, 9, 3, 2)
w_bound = np.sqrt(3 * 9 * 9)
W = np.asarray(np.random.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=img.dtype)

B = np.zeros(2).astype(img.dtype)

# computational graph
x = tf.placeholder(tf.float32, img_shape)

w = tf.Variable(W, tf.float32)
b = tf.Variable(B, tf.float32)

u = tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'VALID') + b
y = tf.nn.sigmoid(u)
o = tf.nn.max_pool(y, ksize=[1, 5, 5, 1], strides = [1, 5, 5, 1], padding = 'VALID')


# initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# evaluate u and y
u_, y_, o_ = sess.run([u, y, o], {x: img.reshape(img_shape)})



# plot the three channels of the image
pylab.figure()
pylab.subplot(1, 4, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()
pylab.subplot(1, 4, 2); pylab.axis('off'); pylab.imshow(img[:, :, 0])
pylab.subplot(1, 4, 3); pylab.axis('off'); pylab.imshow(img[:, :, 1])
pylab.subplot(1, 4, 4); pylab.axis('off'); pylab.imshow(img[:, :, 2])
pylab.savefig('./figures/t7q2_1.png')

# plot original image and first and second components of output
pylab.figure()
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(y_[0, :, :, 0])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(y_[0, :, :, 1])
pylab.savefig('./figures/t7q2_2.png')

pylab.figure()
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(o_[0, :, :, 0])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(o_[0, :, :, 1])
pylab.savefig('./figures/t7q2_3.png')

pylab.show()

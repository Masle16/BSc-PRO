from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def main():
    mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
    
    trainX, testX, trainY, testY = mnist.train.images, mnist.test.images, mnist.train.labels, mnist.test.labels

    plt.figure(1)
    ind = np.random.randint(size=30, low=0, high=50000)

    for i in range(30):
        plt.subplot(3,10,i+1)
        x = trainX[ind[i],:]
        im = x.reshape((28,28))
        img = misc.toimage(im)
        plt.axis('off')
        plt.imshow(img, cmap = plt.cm.gray)

    print(trainY[:30])
    plt.show()


if __name__ == '__main__':
    main()




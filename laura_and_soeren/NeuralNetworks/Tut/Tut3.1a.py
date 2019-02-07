## Design a classifier for a dichotomizer 
import tensorflow as tf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## Definining data and learning parameter
lr = 0.01
iters = 50
SEED = 10
np.random.seed(SEED)


X = np.array([[5,1],[7,3],[3,2],[5,4], [0,0],[-1,-3],[-2,3],[-3,0]])
Y = np.array([0,0,0,0,1,1,1,1])

    # Plot the data and centroid
no_classes = 2
no_data = len(X)
mu = np.zeros((no_classes,2)) # Container for 2 centroids
nc = np.zeros(no_classes)   # Number of elements per class

for p in range(no_data):
    mu[Y[p]] += X[p]        # Y[p] decide index of class, and add responding X
    nc[Y[p]] +=1            # Count out added elements

mu /= nc                    # Decide sum with number of elements
print('centroids: %s'%mu)


plt.figure(1)
plt.plot(X[Y==0, 0], X[Y==0, 1], 'rx', label = 'class 1')
plt.plot(X[Y==1, 0], X[Y==1, 1], 'bx', label = 'class 2')
plt.plot(mu[:,0], mu[:,1], 'o', color = 'black', label = 'centroids')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Data and Centroids')


    # Plot the desision boundry
w = mu[0]-mu[1]
b = 0.5*(np.dot(mu[1],mu[1]) - np.dot(mu[0],mu[0]) )
print('weights: %s and bias: %g'%(w, b))
print('\n')

plt.figure(2)
plt.plot( X[Y==0,0], X[Y==0,1], 'rx', label = 'class 1')
plt.plot( X[Y==1,0], X[Y==1,1], 'bx', label = 'class 1')
plt.plot( mu[:,0], mu[:,1], 'black', linestyle='--', marker='o' )

x1 = np.arange(-4, 5, 0.1)              # Generate points
x2 = np.zeros(len(x1))                  # Container for coresponding x2
for i in range(len(x1)):                
    x2[i] = -(w[0]*x1[i] + b)/w[1]      # Calculate x2
plt.plot(x1, x2, '-', color='black')    # Plot the line
plt.axis('equal')
plt.ylim(-7, 7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Data and Decision Boudnary')

##plt.show()

##Build the graph
    # We already have w and b from before

def test(x):
    u = np.dot(x,w) +b
    if u > 0:
        y = 0
    else:
        y = 1
    return u,y

## Do the loop
for p in range(len(X)):
    out, label = test(X[p])
    print('x: %s'%X[p], 'u: %g'%out, 'y: %d'%label, 'Y: %d'%Y[p])
print('\n')



X_T = [[4, 2], [0, 5], [36/13, 0]]
for p in range(3):
    out, label = test(X_T[p])
    print('x: %s'%X_T[p],' u: %g'%out, 'y:%d'%label)


##plt.show()

    



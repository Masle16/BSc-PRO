#
# Tutorial 3, Question 1
#


import numpy as np
import tensorflow as tf
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

    
# data
X = np.array([[5, 1],[7, 3], [3, 2], [5, 4], [0, 0], [-1, -3], [-2, 3], [-3, 0]])
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# class means
no_classes = 2
no_data = len(X)
mu = np.zeros((no_classes,2))
nc = np.zeros(no_classes)
for p in range(no_data):
    mu[Y[p]] += X[p]
    nc[Y[p]] += 1
mu /= nc
print('centroids: %s'%mu)



# plot data points and centroids 
plt.figure(1)
plt.plot(X[Y==0, 0], X[Y==0, 1], 'rx', label = 'class 1')
plt.plot(X[Y==1, 0], X[Y==1, 1], 'bx', label = 'class 2')
plt.plot(mu[:,0], mu[:,1], 'o', color = 'black', label = 'centroids')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Data and Centroids')
plt.savefig('./figures/t3q1_1.png')

# compute the decision boundary
w = mu[0] - mu[1]
b = 0.5*(np.dot(mu[1],mu[1]) - np.dot(mu[0], mu[0]))
print('weights: %s and bias: %g'%(w, b))
print('\n')

# plot data and the decision boundary
plt.figure(2)
plt.plot(X[Y==0, 0], X[Y==0, 1], 'rx', label = 'class 1')
plt.plot(X[Y==1, 0], X[Y==1, 1], 'bx', label = 'class 2')
plt.plot(mu[:,0], mu[:,1], color = 'black', linestyle='--', marker='o')

x1 = np.arange(-4, 5, 0.1)
x2 = np.zeros(len(x1))
for i in range(len(x1)):
    x2[i] = -(w[0]*x1[i] + b)/w[1]
plt.plot(x1, x2, '-', color='black')
plt.axis('equal')
plt.ylim(-7, 7)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Data and Decision Boudnary')
plt.savefig('./figures/t3q1_2.png')


# find the label on input
def test(x):
    u = np.dot(x, w) + b
    if u > 0:
        y = 1
    else:
        y = 0
    return u, y


# testing on trained patters
for p in range(no_data):
    out, label = test(X[p])
    print('x: %s'%X[p], 'u: %g'%out, 'y: %d'%label)
print('\n')

# testing on new patterns
X_T = [[4, 2], [0, 5], [36/13, 0]]
for p in range(3):
    out, label = test(X_T[p])
    print('x: %s'%X_T[p],' u: %g'%out, 'y:%d'%label)


plt.show()

    

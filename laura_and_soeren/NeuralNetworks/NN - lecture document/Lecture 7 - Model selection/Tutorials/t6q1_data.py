import numpy as np
import tensorflow as tf
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

no_features = 2    

seed = 10
np.random.seed(seed)

# generate training data
X = np.zeros((10*10, no_features))
no_data = 0
for i in np.arange(-1.0, 1.001, 2.0/9.0):
    for j in np.arange(-1.0, 1.001, 2.0/9.0):
        X[no_data] = [i, j]
        no_data += 1

Y = np.zeros((no_data, 1))
Y[:,0] = np.sin(np.pi*X[:,0])*np.cos(2*np.pi*X[:,1])

idx = np.arange(no_data)
np.random.shuffle(idx)
X, Y = X[idx], Y[idx]
Xtrain, Ytrain, Xtest, Ytest = X[:70], Y[:70], X[70:], Y[70:]

# plot the learned function
fig = plt.figure(1)
ax = fig.gca(projection = '3d')
X1 = np.arange(-1, 1, 0.05)
X2 = np.arange(-1, 1, 0.05)
X1,X2 = np.meshgrid(X1,X2)
Z = np.sin(np.pi*X1)*np.cos(2*np.pi*X2)
ax.plot_surface(X1, X2, Z)
#ax.xaxis.set_major_locator(ticker.IndexLocator(base = 0.2, offset=0.0))
#ax.yaxis.set_major_locator(ticker.IndexLocator(base = 0.2, offset=0.0))
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$y$')
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])
plt.savefig('./figures/t6q1_data_1.png')

# plot trained and predicted points
plt.figure(2)
plt.plot(X[:,0], X[:,1], 'b.')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('./figures/t6q1_data_2.png')

fig = plt.figure(3)
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0], X[:,1], Y[:,0], 'b.')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])
#ax.set_title('Targets for Training')
plt.savefig('./figures/t6q1_data_3.png')


# plot trained and predicted points
plt.figure(4)
plt.plot(Xtrain[:,0], Xtrain[:,1], 'b.', label='train')
plt.plot(Xtest[:,0], Xtest[:,1], 'rx', label='test')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('test and train inputs')
plt.legend()
plt.savefig('./figures/t6q1_data_4.png')

fig = plt.figure(5)
ax = fig.gca(projection = '3d')
ax.scatter(Xtrain[:,0], Xtrain[:,1], Ytrain[:,0], 'b.', label='train')
ax.scatter(Xtest[:,0], Xtest[:,1], Ytest[:,0], 'rx', label='test')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.legend()
plt.savefig('./figures/t6q1_data_5.png')

plt.show()

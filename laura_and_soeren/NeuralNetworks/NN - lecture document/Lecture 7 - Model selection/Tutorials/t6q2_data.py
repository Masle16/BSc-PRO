import numpy as np
import tensorflow as tf
import pylab as plt

import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

no_features = 2
no_folds = 5

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

z = []
for fold in range(no_folds):
    start, end = fold*20, (fold+1)*20
    z.append(X[start:end])

z = np.array(z)


# plot trained and predicted points
plt.figure(1)
plt.plot(z[0,:,0], z[0, :,1], '.', label='fold 1')
plt.plot(z[1,:,0], z[1, :,1], '.', label='fold 2')
plt.plot(z[2,:,0], z[2, :,1], '.', label='fold 3')
plt.plot(z[3,:,0], z[3, :,1], '.', label='fold 4')
plt.plot(z[4,:,0], z[4, :,1], '.', label='fold 5')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('data in different folds')
plt.legend()
plt.savefig('./figures/t6q2_data_1.png')

plt.show()

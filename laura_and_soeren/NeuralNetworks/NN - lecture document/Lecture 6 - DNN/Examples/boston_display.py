import numpy as np
from sklearn import datasets


boston = datasets.load_boston()

x = np.array(boston.data)
y = np.array(boston.target)

print(x[:10])
print(y[:10])

train_x = (x - np.mean(x, axis=0))/np.std(x, axis=0) 
train_y = np.array(y)


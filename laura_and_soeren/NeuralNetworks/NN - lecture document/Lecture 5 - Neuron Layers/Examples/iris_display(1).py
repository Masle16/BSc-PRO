from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
iris.data -= np.mean(iris.data, axis=0)

X = iris.data
Y = iris.target

print(X)
print(Y)



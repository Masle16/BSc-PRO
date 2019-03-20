import numpy as np
input = np.asarray([[1,1],[2,2],[3,3]])
x = np.asarray([[[0,0]],[[0,1]]])
x=x.reshape(2,2)
print(np.fliplr(x))
print(input.item(x))

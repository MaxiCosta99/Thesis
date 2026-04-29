import numpy as np
import scipy

a = np.array([1, 2, 3])
b = np.array([6.7, 8, 3])

b = b[a<=4]

print(b)
import numpy as np
import scipy


test = np.array([[3,4,4],[1,2,3]])
test1 = np.array([2,0])
test1 = test1[:,np.newaxis]

print(test*test1)
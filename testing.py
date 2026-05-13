import numpy as np
import scipy
from two_agent import true_A_new

myarr = np.array([[2,3],[4,5],[6,7]])
myarr[0:2, :] = 0

print(myarr)
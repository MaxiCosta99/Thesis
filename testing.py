import numpy as np
import scipy
from two_agent import true_A_new

y = true_A_new(0,1,1)

x = np.load("inits/ta_learnedA.npy")

terms = "jklo -> ko"

for i in range(12):

    print(np.einsum(terms, y[1][i])-np.round(np.einsum(terms, x[i]),2))
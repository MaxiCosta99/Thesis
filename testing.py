import numpy as np
import scipy


def compute_cov_dir(x):
    x = x.reshape(-1)
    x_norm = x/np.sum(x)
    x_norm = x_norm[:,np.newaxis]

    y = (np.diag(x_norm.reshape(-1))-(x_norm @ x_norm.T))/(np.sum(x)+1)

    E = scipy.linalg.eigh(y, eigvals_only = True)
    
    assert np.all(E > -1e-8)

    return y


A = np.random.uniform(0,1,(5,4,2))


def compute_cov_A(A):
    """
    A is of shape (num_obs, ...)
    """
    
    A = A.reshape(A.shape[0],-1)
    print(A.shape)
    
    cov_mat_sum = 0

    for a in A.T:
        cov_mat = compute_cov_dir(a)
        cov_mat_sum = cov_mat_sum + cov_mat

    return cov_mat_sum



cov_mat_sum = compute_cov_A(A)

print(cov_mat_sum)



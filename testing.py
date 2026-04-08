import numpy as np

num_acts = 4
sfes = np.array([0,1,2,4,4])
us = np.array([0,0,1,1,2])

u_imagined = []
H = []

for u in range(num_acts):
    temp = sfes[us==u]
    if len(temp) > 0:
        sfes_mean = np.mean(temp)
        u_imagined.append(u)
        H.append(sfes_mean)


arr = np.array([[0,1,2,4],[0,2,3,4]])
#test = np.broadcast_to(arr, (2,4,4))
#test = np.broadcast_to(test, (2,2,4,4))
#test = np.repeat(arr[np.newaxis,:,:], axis = 0, repeats = 2)
#test = np.repeat(test[:,:,np.newaxis,:], axis = 2, repeats = 4)

arr = np.array([1,2,3])
arr2 = np.array([4,5,6])
arrlist = [arr, arr2]

newlist = arrlist.copy()

print(newlist)
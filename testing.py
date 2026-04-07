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

print(np.min(us))

print(u_imagined)
print(H)
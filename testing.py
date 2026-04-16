import numpy as np

test = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])

test1 = np.reshape(test, (10), order = "C")
test2 = np.reshape(test, (10), order = "F")

print(test)
print(test1)
print(test2)
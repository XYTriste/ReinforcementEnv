import numpy as np
s = np.array([1, 3, 4, 5])
a = 1
r = 0.8
s_prime = np.array([4, 5, 1, 2])
print(np.hstack((s, [a, r], s_prime)))
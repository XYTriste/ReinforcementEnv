import matplotlib.pyplot as plt
import numpy as np
import time

plt.axis([0, 100, 0, 1])
plt.ion()

for i in range(100):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

plt.ioff()
plt.show()

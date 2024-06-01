import random

import numpy as np
import matplotlib.pyplot as plt
import ast
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

data = {}
with open('data/invalid action position.txt', 'r') as file_obj:
    lines = file_obj.readlines()
    for line in lines:
        state, action = line.strip().split(',')
        state = int(state)
        action = int(action)
        if (state, action) not in data:
            data[(state, action)] = 1
        else:
            data[(state, action)] += 1
x = [key[0] for key in data.keys()]
y = [key[1] for key in data.keys()]
sizes = [value for value in data.values()]
sizes = np.array(sizes)

norm = Normalize(vmin=sizes.min(), vmax=sizes.max())
sizes_normalized = 100 + 500 * norm(sizes)

fig, ax = plt.subplots()
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
for i in range(len(x)):
    ax.scatter(x[i], y[i], label='({}, {})'.format(x[i], y[i]), s=sizes_normalized[i])
# 绘制散点图，设置散点大小为键值大小
# plt.scatter(x, y, s=sizes_normalized)
plt.xlabel('state')
plt.ylabel('action')
plt.title('Scatter Plot with Different Sizes')
plt.show()
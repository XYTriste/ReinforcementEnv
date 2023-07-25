import ast
import matplotlib.pyplot as plt
f1 = open('./data/use_A.txt', 'r')
# f2 = open('./data/use_A_before.txt', 'r')

average_data_A = 0
average_data_B = 0
use_A = ast.literal_eval(f1.readline())
# unuse_A = ast.literal_eval(f2.readline())
plot_A = []
plot_B = []
for i in range(len(use_A)):
    average_data_A = average_data_A + (use_A[i] - average_data_A)/(i + 1)
    plot_A.append(average_data_A)

    # average_data_B = average_data_B + (unuse_A[i] - average_data_B)/(i + 1)
    # plot_B.append(average_data_B)
plt.plot(plot_A, color='r', label="use A")
# plt.plot(plot_B, color='b', label="unuse A")
plt.show()
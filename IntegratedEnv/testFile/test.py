import ast
f1 = open('../data/PPO_Breakout-v5_All Process_1000r-1023642-_RND_23_07_09_12.txt', 'r')
list1 = ast.literal_eval(f1.readline())
print(max(list1))
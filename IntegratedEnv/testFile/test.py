import ast
f1 = open('../data/PPO_MontezumaRevenge-v5_All Process_2000r-2047020-_RND_23_07_09_16.txt', 'r')
list1 = ast.literal_eval(f1.readline())
print(max(list1))
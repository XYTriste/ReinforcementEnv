import ast
f1 = open('../data/PPO_MontezumaRevenge-v5_All Process_1000r-1685445157-_RND_23_07_09_08.txt', 'r')
list1 = ast.literal_eval(f1.readline())
print(sum(list1))
import ast
f1 = open('../data/PPO_MontezumaRevenge-v5_All Process_final_RND_23_07_08_21.txt', 'r')
list1 = ast.literal_eval(f1.readline())
print(len(list1))
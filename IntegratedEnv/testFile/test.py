import ast
f1 = open('../data/PPO_Breakout-v5_All Process_final-10239645-_RND_23_07_09_17.txt', 'r')
list1 = ast.literal_eval(f1.readline())
print(max(list1))
'''
48476 343147
62263
121174

33953  224553

'''
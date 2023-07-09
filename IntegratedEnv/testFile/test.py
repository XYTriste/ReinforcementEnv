import ast
f1 = open('../data/PPO/PPO_Breakout-v5_All Process_2000r-2047643-_RND_23_07_09_22.txt', 'r')
list1 = ast.literal_eval(f1.readline())
dic = {}
for i in range(len(list1)):
    if list1[i] in dic.keys():
        dic[list1[i]] += 1
    else:
        dic[list1[i]] = 1
print(dic)
'''
48476 343147
62263
121174

33953  224553

'''
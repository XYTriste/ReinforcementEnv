import ast
f1 = open('../data/PPO/PPO_Breakout-v5_All Process_16000r-16257228-_RND_23_07_10_05.txt', 'r')
f2 = open('../data/PPO/PPO_Breakout-v5_All Process_2000r-2047643-_RND_23_07_09_22.txt', 'r')
list1 = ast.literal_eval(f1.readline())
list2 = ast.literal_eval(f2.readline())
dic1 = {}
dic2 = {}
for i in range(len(list1)):
    if list1[i] in dic1.keys():
        dic1[list1[i]] += 1
    else:
        dic1[list1[i]] = 1
for i in range(len(list2)):
    if list2[i] in dic2.keys():
        dic2[list2[i]] += 1
    else:
        dic2[list2[i]] = 1
print(dic1)
print(dic2)
'''
48476 343147
62263
121174

33953  224553

'''
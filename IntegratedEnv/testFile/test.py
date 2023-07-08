import ast
f1 = open('../data/All Process__23_07_08_13.txt', 'r')
list1 = ast.literal_eval(f1.readline())
print(sum(list1))
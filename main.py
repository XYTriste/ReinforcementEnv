import matplotlib.pyplot as plt

# 创建Figure对象和Axes对象
fig, ax = plt.subplots()

# 绘制第一条折线
x1 = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
ax.plot(x1, y1, label='Line 1')

# 绘制第二条折线
x2 = [1, 2, 3, 4, 5]
y2 = [1, 3, 5, 7, 9]
ax.plot(x2, y2, label='Line 2')

# 添加图例
ax.legend()

plt.savefig("./difference.png")
# 显示图形
plt.show()

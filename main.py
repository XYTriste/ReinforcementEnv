import numpy as np
import matplotlib.pyplot as plt

# 创建初始的空曲线
x = []
y = []

# 创建图形对象
fig, ax = plt.subplots()

# 创建初始的曲线对象
line, = ax.plot(x, y)


# 更新曲线的函数
def update_curve(new_data):
    # 添加新数据
    x.append(len(x))
    y.append(new_data)

    # 更新曲线的数据
    line.set_data(x, y)

    # 重新调整曲线的范围
    ax.relim()
    ax.autoscale_view()

    # 重新绘制图形
    fig.canvas.draw()


# 添加新数据并更新曲线
for i in range(10):
    new_data = np.random.rand()
    update_curve(new_data)

# 显示图形
plt.show()

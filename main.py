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
# def update_curve(new_data):
#     # 添加新数据
#     x.append(len(x))
#     y.append(new_data)
#
#     # 更新曲线的数据
#     line.set_data(x, y)
#
#     # 重新调整曲线的范围
#     ax.relim()
#     ax.autoscale_view()
#
#     # 重新绘制图形
#     fig.canvas.draw()
#
#
# # 添加新数据并更新曲线
# for i in range(10):
#     new_data = np.random.rand()
#     update_curve(new_data)
#
# # 显示图形
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成随机数据
# data = np.random.rand(10, 10)
#
# # 绘制热力图
# plt.imshow(data, cmap='hot', interpolation='nearest')
#
# # 添加颜色条
# plt.colorbar()
#
# # 显示图形
# plt.show()

import gymnasium as gym

env = gym.make("Blackjack-v1", render_mode="rgb_array")


def easy_policy(player_state):
    if player_state < 15:
        return 1
    else:
        return 0


rounds = 10000
win_rounds = 0
lose_rounds = 0

for i in range(rounds):
    state, _ = env.reset()
    done = False

    while not done:
        action = easy_policy(state[0])
        state, reward, done, _, _ = env.step(action)
        if done:
            if reward > 0:
                win_rounds += 1
            elif reward < 0:
                lose_rounds += 1

print("win rate:{:3f}  not lose rate:{:3f}   lose rate:{:3f}".format((win_rounds / rounds),
                                                                     (rounds - lose_rounds) / rounds,
                                                                     (lose_rounds / rounds)))

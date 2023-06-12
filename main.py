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

# import gymnasium
# import keyboard
#
# # 创建Montezuma's Revenge环境
# env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode="human")
#
# # 重置环境并获取初始观察
# observation, _ = env.reset()
#
#
# # 执行1000个动作
# while True:
#     # 随机选择一个动作
#     # action = env.action_space.sample()
#
#
#
#
#     # 执行选定的动作，并获取下一个观察、奖励、终止标志和额外信息
#     next_observation, reward, done, info, _ = env.step(action)
#
#     # 在控制台打印当前观察和奖励
#     print('Observation:', next_observation)
#     print('Reward:', reward)
#
#     # 如果游戏结束，重置环境
#     if done:
#         observation, _ = env.reset()
#     else:
#         observation = next_observation
#
# # 关闭环境
# env.close()

# import cv2
# import numpy as np
# import gymnasium as gym
#
# def preprocess_observation(observation):
#     # 将原始观测从RGB图像转换为灰度图像
#     gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
#     # 对灰度图像进行裁剪，截取游戏关键区域
#     cropped = gray[34:194, :]
#     # 调整图像大小为所需的尺寸
#     resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
#     # 将像素值归一化到[0, 1]范围
#     normalized = resized / 255.0
#     # 返回处理后的状态特征
#     return normalized
#
# # 示例使用方法
# env = gym.make("ALE/MontezumaRevenge-v5")
# observation, _ = env.reset()
# state = preprocess_observation(observation)
# print(state)
import torch
state = torch.randn(32, 4, 84, 84)  # (batch_size, color_channel, img_height,img_width)
state_size = state.size()
print(state_size)
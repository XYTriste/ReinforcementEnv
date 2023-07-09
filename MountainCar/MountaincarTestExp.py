import gymnasium
import numpy as np


def attention_mechanism(observation):
    # 提取智能体当前位置和旗子位置
    agent_pos = observation[0]
    flag_pos = 0.5  # 假设旗子始终在这个固定位置上

    # 计算智能体和旗子之间的距离
    distance = abs(agent_pos - flag_pos)

    # 计算注意力权重，使得智能体更关注距离较近的情况
    attention_weight = 1 / (distance + 1e-6)  # 避免除零错误

    return attention_weight
def convert_to_probability_distribution(values):
    values = np.array(values)

    # 将值转换为非负数
    values = values - np.min(values)

    # 计算总和
    total = np.sum(values)

    # 计算概率分布
    probabilities = values / total

    # 确保概率和为1
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

# 示例使用
values = [1, 2, 3, 4, 5]
probabilities = convert_to_probability_distribution(values)
print(probabilities)


env = gymnasium.make('MountainCar-v0', render_mode="human")

observation, _ = env.reset()

done = False
total_reward = 0

while not done:
    #env.render()

    # 应用注意力机制
    attention_weight = attention_mechanism(observation)
    prob = convert_to_probability_distribution([attention_weight, 0, 1 - attention_weight])

    # 根据注意力权重选择动作
    action = np.random.choice(env.action_space.n, p=prob)
    print("{}".format("left" if action == 0 else "right"))

    observation, reward, done, info, _ = env.step(action)
    total_reward += reward

print("Total reward:", total_reward)

env.close()

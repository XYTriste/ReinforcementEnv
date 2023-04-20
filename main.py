import gym
import numpy as np
from gym import spaces

env = gym.make('CarRacing-v2', render_mode="human")
obs = env.reset()

# 定义动作空间和状态空间的大小
n_actions = 3
state_shape = env.observation_space.shape

# 定义策略和价值函数的初始值
policy = np.ones((state_shape[0], state_shape[1], n_actions)) / n_actions
Q = np.zeros((state_shape[0], state_shape[1], n_actions))

# 定义超参数
gamma = 0.99
alpha = 0.1
epsilon = 0.1
n_episodes = 1000

# 定义一个函数，用于从策略中选择动作
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(policy[state[0], state[1]])

# 进行蒙特卡洛算法训练
for i_episode in range(n_episodes):
    obs = env.reset()
    done = False
    episode = []

    # 收集一次完整的轨迹
    while not done:
        state = (int(obs[0]*100), int(obs[1]*100))
        action = choose_action(state, epsilon)
        obs_next, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        obs = obs_next

    # 更新策略和价值函数
    G = 0
    for t in range(len(episode)-1, -1, -1):
        state, action, reward = episode[t]
        G = gamma * G + reward
        if not (state, action) in [(episode[i][0], episode[i][1]) for i in range(t)]:
            Q[state[0], state[1], action] += alpha * (G - Q[state[0], state[1], action])
            best_action = np.argmax(Q[state[0], state[1]])
            for a in range(n_actions):
                if a == best_action:
                    policy[state[0], state[1], a] = 1 - epsilon + epsilon/n_actions
                else:
                    policy[state[0], state[1], a] = epsilon/n_actions

# 进行测试
obs = env.reset()
done = False
total_reward = 0
while not done:
    state = (int(obs[0]*100), int(obs[1]*100))
    action = choose_action(state, 0)
    obs, reward, done, info = env.step(action)
    total_reward += reward
env.close()

print('Total reward:', total_reward)

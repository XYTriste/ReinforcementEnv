import math

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class AlphaMCControl:
    def __init__(self, env, alpha, epsilon, gamma):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = {}  # 存储状态-动作值函数
        self.win_rate = np.zeros((32, 11, 2))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            # 探索，随机选择一个动作
            action = self.env.action_space.sample()
        else:
            # 利用，选择具有最高Q值的动作
            if state in self.Q:
                q_values = self.Q[state]
                action = np.argmax(q_values)
            else:
                # 如果状态尚未探索过，则随机选择一个动作
                action = self.env.action_space.sample()
        return action

    def choose_action_greedy(self, state):
        return np.argmax(self.Q[state])

    def choose_action_easy(self, state):
        if state[0] < 15:
            return 1
        else:
            return 0

    def update_Q(self, episode):
        returns = 0
        for state, action, reward in reversed(episode):
            returns = self.gamma * returns + reward
            if state in self.Q:
                q_values = self.Q[state]
                q_values[action] = (1 - self.alpha) * q_values[action] + self.alpha * returns
            else:
                # 如果状态尚未探索过，则将其添加到Q表中
                self.Q[state] = np.zeros(self.env.action_space.n)
                self.Q[state][action] = self.alpha * returns

    def plot_reward(self, reward_list, window, end=False):
        plt.figure(window)
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title("Monte Carlo on BlackJack-v0")
        list_t = [np.mean(reward_list[:i]) for i in range(len(reward_list))]
        plt.plot(np.array(list_t))
        if end:
            plt.show()
        else:
            plt.pause(0.001)

    def train(self, num_episodes):
        reward_list = []
        train_rounds = 0
        player_win, dealer_win, draw = 0, 0, 0
        while train_rounds < 30000000 or (train_rounds < 30000000 and player_win / train_rounds < 44.0):
            if train_rounds % 100000 == 0 and train_rounds > 0:
                print("Train {} rounds.  win rate:{:4f}%".format(train_rounds, player_win / train_rounds))
            state, _ = self.env.reset()
            start_state = state
            done = False
            episode_reward = 0
            episode_data = []  # 用于存储每个episode的状态、动作和奖励

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                state = next_state
                episode_reward += reward

            if episode_reward >= 0:
                if episode_reward > 0:
                    player_win += 1
                elif episode_reward == 0:
                    draw += 1
                else:
                    dealer_win += 1
                self.win_rate[start_state[0]][start_state[1]][1 if start_state[2] else 0] += 1
            train_rounds += 1
            reward_list.append(episode_reward)
            # self.plot_reward(reward_list, 1)
            self.update_Q(episode_data)

    def test(self, num_episodes):
        player_win_count = 0
        dealer_win_count = 0
        reward_list = []
        principal = 2000
        for episode in range(num_episodes):
            state, _ = self.env.reset()

            done = False
            episode_reward = 0

            prob = self.win_rate[state[0]][state[1]][1 if state[2] else 0]
            bet = math.floor((prob / 50000) * 10)
            bet = max(bet, 1)
            bet = min(bet, 100)

            while not done:
                action = self.choose_action_greedy(state)

                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    if reward > 0:
                        player_win_count += 1
                        principal += bet
                    elif reward < 0:
                        dealer_win_count += 1
                        principal -= bet
            reward_list.append(episode_reward)
            # self.plot_reward(reward_list, 1)
        print("Play with dealer {} rounds. Player win rate:{:5f} "
              "Player not lose rate: {:5f}"
              "  dealer win rate: {:5f}   principal:{}".format(num_episodes, player_win_count / num_episodes,
                                                (num_episodes - dealer_win_count) / num_episodes,
                                                dealer_win_count / num_episodes, principal))


# 使用示例
env = gym.make('Blackjack-v1', render_mode="rgb_array")
agent = AlphaMCControl(env, alpha=0.002, epsilon=0.1, gamma=0.99)
agent.train(num_episodes=50000)
agent.test(num_episodes=10000)
for i in range(2, 22):
    for j in range(1, 11):
        print("玩家 {} 点， 庄家{} 点.最优行为是:{}".format(i, j, "抽牌" if np.argmax(agent.Q[i, j, False]) else "不抽牌"))
action_array = []
for i in range(21, 10, -1):
    lis = []
    for j in range(2, 11):
        if (i, j, False) in agent.Q:
            lis.append(np.argmax(agent.Q[(i, j, False)]))
        else:
            lis.append(0)
    action_array.append(lis)
action_values = np.array(action_array)
fig, ax = plt.subplots(figsize=(6, 6))
colors = [(0, 'red'),  (1, 'green')]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
heatmap = ax.imshow(action_values, cmap=cmap, vmin=0, vmax=1, extent=[2, 10, 11, 21])

plt.xticks(range(2, 11))
plt.yticks(range(11, 22))
# 自定义标签和标题
# ax._xlabel("Dealer's Card Value")
# ax._ylabel("Player's Sum")
# ax._title("Action Value Function")
# ax.grid(True)
# 添加颜色条
cbar = plt.colorbar(heatmap)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["stick", "hit"])

# 显示热力图
plt.show()
# best_actions = np.zeros((32, 11))

# 根据最佳行为设置相应位置为0或1
# for i in range(32):
#     for j in range(11):
#         best_actions[i, j] = np.argmax(agent.Q[i, j])
#
# # best_actions = best_actions[10:22, :]
# # 绘制热力图
# plt.imshow(best_actions[:, 1:], cmap='coolwarm', aspect='auto')
#
# # 添加坐标轴标签
# plt.xlabel("Dealer state")
# plt.ylabel("Player state")
#
# # 设置坐标轴刻度
# plt.xticks(range(1, 11))
# plt.yticks(range(1, 32))
#
# # 添加颜色条
# cbar = plt.colorbar()
# cbar.set_ticks([0, 1])
# cbar.set_ticklabels(["stick", "hit"])
#
# plt.grid(True)
#
# # 显示图形
# plt.show()

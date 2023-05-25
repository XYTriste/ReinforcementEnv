import gym
import numpy as np
import matplotlib.pyplot as plt


class AlphaMCControl:
    def __init__(self, env, alpha, epsilon, gamma):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = {}  # 存储状态-动作值函数

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
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_data = []  # 用于存储每个episode的状态、动作和奖励

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                state = next_state
                episode_reward += reward

            reward_list.append(episode_reward)
            # self.plot_reward(reward_list, 1)
            self.update_Q(episode_data)

    def test(self, num_episodes):
        player_win_count = 0
        dealer_win_count = 0
        reward_list = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.choose_action_greedy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    if reward > 0:
                        player_win_count += 1
                    elif reward < 0:
                        dealer_win_count += 1
            reward_list.append(episode_reward)
            self.plot_reward(reward_list, 1)
        print("Play with dealer {} rounds. Player win rate:{:5f} "
              "Player not lose rate: {:5f}"
              "  dealer win rate: {:5f}".format(num_episodes, player_win_count / num_episodes,
                                                (num_episodes - dealer_win_count) / num_episodes,
                                                dealer_win_count / num_episodes))


# 使用示例
env = gym.make('Blackjack-v1', render_mode="rgb_array")
agent = AlphaMCControl(env, alpha=0.002, epsilon=0.1, gamma=0.9)
agent.train(num_episodes=50000)
agent.test(num_episodes=2000)
plt.show()

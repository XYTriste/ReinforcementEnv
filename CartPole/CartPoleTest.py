import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0", render_mode="rgb_array")
env = env.unwrapped
N_ACTIONS = 2
N_STATES = env.observation_space.shape[0]

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100  # 目标网络的更新速率，100指的是每更新当前网络100次则更新一次目标网络
MEMORY_CAPACITY = 2000
EPISODE = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # 网络权重参数初始化
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal(0, 0.1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            actions_value = self.out(x)

            return actions_value


class DQN:
    def __init__(self):
        self.main_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0  # 已训练的次数计数器
        self.memory_counter = 0  # 已保存的经验样本数量计数器
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 经验样本缓存池
        # 第一维是缓存池的大小，第二维是每个样本的大小。样本包含当前状态(4)，即时奖励(1)，动作(1)，后续状态(4)
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 在x的第0维添加一个维度，达到升维的效果

        if np.random.uniform(0, 1) < EPSILON:
            actions_value = self.main_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()  # 在actions_value的第1维上计算最大值
            # 最大值返回的是一个元组，分别表示最大值及其下标。因此获取其下标并传递给cpu最后以numpy的形式返回
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)

        return action

    def store_transition(self, s, a, r, s_prime):
        transition = np.hstack((s, [a, r], s_prime))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, :N_STATES])
        batch_a = torch.LongTensor(batch_memory[:, N_STATES: N_STATES + 1].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, N_STATES + 1, N_STATES + 2])
        batch_s_prime = torch.FloatTensor(batch_memory[:, -N_STATES:])

        q = self.main_net(batch_s).gather(1, batch_a)  # 主网络对输入的一批状态，计算每个状态下所有的行为价值
        # 并从第 1 维中取出一批行为的价值。返回一个1维张量q，包含了这批状态-行为对的行为价值。
        q_target = self.target_net(batch_s_prime).detach()  # 在目标网络中计算后续状态的所有可能的行为价值
        # detach函数将其从计算图中分离出来，避免在反向传播过程中产生新的梯度。返回的是一个新的张量，但是它与原始张量共享
        # 底层存储
        y = batch_r + GAMMA * q_target.max(1)[0].view(BATCH_SIZE, 1)
        # 类似于TD target的计算，只不过是向量形式的。q_target.max(1)取出每个状态下最大的行为价值及其索引。
        # 然后q_target.max(1)[0]就会取出所有的状态下的最大行为价值。最后调用view方法来将张量重塑为可以计算的形状。
        loss = self.loss_func(q, y)

        self.optimizer.zero_grad()  # 将所有可学习的参数的梯度清零，否则梯度会累加
        loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新参数


if __name__ == '__main__':
    dqn = DQN()

    plot_x_data, plot_y_data = [], []
    for i in range(500):
        state = env.reset()
        episode_reward = 0
        while True:
            action = dqn.choose_action(state)

            s_prime, r, done, _ = env.step(action)

            x, x_dot, theta, theta_dot = s_prime
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(state, action, r, s_prime)

            episode_reward += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Episode: ', i,
                          '| Episode_reward: ', round(episode_reward, 2))

            if done:
                break
            s = s_prime
        plot_x_data.append(i)
        plot_y_data.append(episode_reward)
        plt.plot(plot_x_data, plot_y_data)
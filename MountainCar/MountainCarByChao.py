import copy
import random
import gym
import numpy as np
from collections import deque
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
import argparse


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 队列,先进先出

    # 将数据加入buffer
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 从buffer中采样数据,数量为batch_size
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前buffer中数据的数量
    def size(self):
        return len(self.buffer)


def define_args():
    parser = argparse.ArgumentParser(description='DQN parametes settings')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the net.')
    parser.add_argument('--num_episodes', type=int, default=1000, help='the num of train epochs')
    parser.add_argument('--seed', type=int, default=21, metavar='S', help='Random seed.')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='S', help='the discount rate')
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='S', help='the epsilon rate')
    parser.add_argument('--target_update', type=float, default=100, metavar='S', help='the frequency of the target net')
    parser.add_argument('--buffer_size', type=float, default=10000, metavar='S', help='the size of the buffer')
    parser.add_argument('--minimal_size', type=float, default=256, metavar='S', help='the minimal size of the learning')
    parser.add_argument('--env_name', type=str, default="MountainCar-v0", metavar='S',
                        help='the name of the environment')
    args = parser.parse_args()
    return args


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, s):
        x = self.fc1(s)
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x))
        x = self.fc4(self.relu(x))
        return x


class DQN:
    def __init__(self, args, epsilon):
        self.args = args
        self.hidden_dim = 128
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = args.target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.num_episodes = args.num_episodes
        self.minimal_size = args.minimal_size
        self.l1lossfn = nn.L1Loss()
        self.env = gym.make(args.env_name, )

        # 探索利用切换阈值
        self.threshold = 1
        random.seed(args.seed)
        np.random.seed(args.seed)
        # self.env.seed(args.seed)
        torch.manual_seed(args.seed)

        self.replay_buffer = ReplayBuffer(args.buffer_size)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # q_网络
        self.q_net = Qnet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)

        # 目标网络
        self.target_q_net = Qnet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # 优化器
        self.optimizer = Adam(self.q_net.parameters(), lr=self.lr)

    def init_model(self):
        """
        初始化模型参数
        :return:
        """
        # self.q_net.load_state_dict(self.init_q.state_dict())
        #
        # self.target_q_net.load_state_dict(self.init_q.state_dict())
        # self.target_q_net.eval()

    def select_action(self, state):  # epsilon-贪婪策略采取动作
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # print(self.q_net(state))
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition):
        states = torch.tensor(transition["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition["dones"], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q value
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD error
        loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

        return loss.item()

    def run(self, modeld="dqn"):
        self.modeld = modeld
        return_list = []
        step = 0
        for i in range(10):
            with tqdm(total=int(self.num_episodes / 10), desc=f'Iteration {i}') as pbar:
                for episode in range(self.num_episodes // 10):
                    episode_return = 0
                    state, _ = self.env.reset()

                    time_step = 0
                    episode_loss_sum = 0
                    episode_loss_count = 0
                    while True and time_step < 200:
                        action = self.select_action(state)
                        # print(action)
                        next_state, reward, done, _, _ = self.env.step(action)
                        episode_return += reward
                        self.replay_buffer.add(state, action, reward, next_state, done)
                        if self.replay_buffer.size() > self.minimal_size and step % 5 == 0:
                            s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)
                            transitions = {"states": s, "actions": a, "rewards": r, "next_states": s_, "dones": d}
                            episode_loss_sum += self.update(transitions)
                            episode_loss_count += 1
                        state = next_state
                        step += 1
                        time_step += 1
                        if done:
                            print("episode average loss:{}".format(episode_loss_sum / episode_loss_count))
                            break
                    return_list.append(episode_return)
                    self.plot_reward(return_list, 1)
                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{self.num_episodes / 10 * i + episode + 1}",
                                "return": f"{np.mean(return_list[-10:]):3f}"
                            }
                        )
                    pbar.update(1)

    def plot_reward(self, reward_list, window, end=False):
        plt.figure(window)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(self.args.env_name))
        list_t = [np.mean(reward_list[:i]) for i in range(len(reward_list))]
        plt.plot(np.array(list_t))
        if end:
            plt.show()
        else:
            plt.pause(0.001)


if __name__ == '__main__':
    args = define_args()
    model = DQN(args, args.epsilon)
    model.init_model()  # 初始化
    rt_1 = model.run()

# -*- coding = utf-8 -*-
# @time: 5/26/2023 9:17 PM
# Author: Yu Xia
# @File: Algorithm.py
# @software: PyCharm
import numpy as np
import torch

from Network import *
from Tools import *


class DQN:
    def __init__(self, args: SetupArgs, INPUT_DIM=2, HIDDEN_DIM=128, OUTPUT_DIM=3, HIDDEN_DIM_NUM=3,
                 SIZEOF_EVERY_MEMORY=7):
        self.NAME = "DQN"
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.INPUT_DIM = INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = OUTPUT_DIM  # 输出层的大小
        self.HIDDEN_DIM_NUM = HIDDEN_DIM_NUM  # 隐藏层的数量
        self.SIZEOF_EVERY_MEMORY = SIZEOF_EVERY_MEMORY  # 经验回放的样本大小，使用不同环境时应进行修改。
        self.MEMORY_SHAPE = (3, 1, 1, 3)  # 经验回放的样本中s, a, r, s_prime所占大小，使用不同环境时应进行修改。

        self.TARGET_REPLACE_ITER = 100
        self.MEMORY_CAPACITY = 10000
        self.LR = 0.001
        self.BATCH_SIZE = 128

        self.epsilon = self.args.epsilon
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.learn_frequency = 0  # 记录执行了多少次step方法，控制经验回放的速率

        self.main_net = Net(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.HIDDEN_DIM_NUM).to(self.device)
        self.target_net = Net(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.HIDDEN_DIM_NUM).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.memory = np.zeros((self.MEMORY_CAPACITY, self.SIZEOF_EVERY_MEMORY))  # 经验回放缓冲区
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.LR)  # 网络参数优化
        self.loss_func = nn.MSELoss()  # 损失函数，默认使用均方误差损失函数

    @torch.no_grad()
    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.OUTPUT_DIM)
        else:
            action = self.main_net(state).argmax().item()

        return action

    def store_transition(self, s, a, r, s_prime, done):
        transition = np.hstack((s, [a, r], s_prime, [done]))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def step(self, s, a, r, s_prime, done) -> float:
        if self.epsilon > 0.01:
            self.epsilon *= 0.99

        self.store_transition(s, a, r, s_prime, done)
        self.learn_frequency += 1
        if self.memory_counter > self.BATCH_SIZE and self.learn_frequency % 5 == 0:
            loss = self.learn()
            return loss
        return 0.0

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]

        batch_s = torch.FloatTensor(batch_memory[:, : self.MEMORY_SHAPE[0]]).to(self.device)
        batch_a = torch.LongTensor(
            batch_memory[:, self.MEMORY_SHAPE[0]: self.MEMORY_SHAPE[0] + 1].astype(int)).to(self.device)
        batch_r = torch.FloatTensor(batch_memory[:, self.MEMORY_SHAPE[0] + 1: self.MEMORY_SHAPE[0] + 2]).to(self.device)
        batch_s_prime = torch.FloatTensor(
            batch_memory[:, self.MEMORY_SHAPE[0] + 2: self.MEMORY_SHAPE[0] + 5]).to(self.device)
        batch_done = torch.LongTensor(batch_memory[:, -1:]).to(self.device)

        estimated_q = self.main_net(batch_s).gather(1, batch_a)
        q_target = self.target_net(batch_s_prime).detach()

        y = batch_r + self.args.gamma * q_target.max(1)[0].view(self.BATCH_SIZE, 1) * (1 - batch_done)

        loss = self.loss_func(estimated_q, y)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


class Actor_Critic:
    def __init__(self, args: SetupArgs):
        self.NAME = "Actor Critic"
        self.args = args

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor = Actor(3, 128, 2).to(self.device)
        self.critic = Critic(3, 128, 2).to(self.device)
        self.gamma = self.args.gamma
        self.LR = self.args.lr

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.LR)

        self.critic_loss_func = nn.MSELoss()

    def select_action(self, state) -> int:
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action_probs = self.actor(state)
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def update(self, state, action: int, reward: float, next_state, done: bool):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0).to(self.device)

        # Compute TD error
        state_value = self.critic(state).squeeze(0)[action]
        next_state_value = self.critic(next_state).squeeze(0).max()
        td_error = reward + self.args.gamma * next_state_value - state_value

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss = self.critic_loss_func(state_value, reward + self.args.gamma * next_state_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        self.actor_optimizer.zero_grad()
        action_probs = self.actor(state)
        log_prob = torch.log(action_probs.squeeze(0)[action])
        actor_loss = -log_prob * td_error.item()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def step(self, s, a, r, s_prime, done):
        return self.update(s, a, r, s_prime, done)

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
    def __init__(self, args: SetupArgs):
        self.NAME = "DQN"
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.INPUT_DIM = 2  # 输入层的大小
        self.HIDDEN_DIM = 128  # 隐藏层的大小
        self.OUTPUT_DIM = 3  # 输出层的大小
        self.HIDDEN_DIM_NUM = 3  # 隐藏层的数量
        self.SIZEOF_EVERY_MEMORY = 7  # 经验回放的样本大小，使用不同环境时应进行修改。
        self.MEMORY_SHAPE = (2, 1, 1, 2)  # 经验回放的样本中s, a, r, s_prime所占大小，使用不同环境时应进行修改。

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
            batch_memory[:, self.MEMORY_SHAPE[0] + 2: self.MEMORY_SHAPE[0] + 4]).to(self.device)
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


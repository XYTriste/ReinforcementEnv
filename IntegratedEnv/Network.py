# -*- coding = utf-8 -*-
# @time: 5/26/2023 9:05 PM
# Author: Yu Xia
# @File: Network.py
# @software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, HIDDEN_LAYERS_NUM=1):
        super(Net, self).__init__()
        self.HIDDEN_LAYERS_NUM = HIDDEN_LAYERS_NUM

        self.input_layer = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.hidden_layers = nn.ModuleList()
        for i in range(HIDDEN_LAYERS_NUM):
            layer = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.hidden_layers.append(layer)
        self.output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNDNetwork(nn.Module):
    """
    该类基于随机网络蒸馏(Random Network Distillation, RND)的思想，基于预测误差给予智能体一定的内在奖励。
    通常用于鼓励智能体更多的探索新的状态。
    """

    def __init__(self):
        super(RNDNetwork, self).__init__()
        self.INPUT_DIM = 2  # 输入层的大小
        self.HIDDEN_DIM = 128  # 隐藏层的大小
        self.OUTPUT_DIM = 3  # 输出层的大小
        self.HIDDEN_DIM_NUM = 1
        self.LR = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor = Net(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.HIDDEN_DIM_NUM).to(self.device)
        self.target = Net(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.HIDDEN_DIM_NUM).to(self.device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    def forward(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        predict = self.predictor(state)
        target = self.target(state)
        return predict, target

    def update_parameters(self, predict, target):
        loss = self.loss_func(predict, target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.predictor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def get_intrinsic_reward(self, predict, target):
        return self.update_parameters(predict, target)

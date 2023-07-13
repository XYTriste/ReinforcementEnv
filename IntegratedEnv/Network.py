# -*- coding = utf-8 -*-
# @time: 5/26/2023 9:05 PM
# Author: Yu Xia
# @File: Network.py
# @software: PyCharm
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms

torch.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  # 如果有GPU和cuda
# ，数据将转移到GPU执行
torch.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


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


class CNN(nn.Module):
    def __init__(self, INPUT_CHANNELS, OUTPUT_CHANNELS):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, OUTPUT_CHANNELS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape((-1, 7 * 7 * 64))
        # x = x.view(x.size(0), -1)
        # x = x.flatten()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SelfAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttentionModel, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)

        attn_weights = F.softmax(q @ k.transpose(-2, -1) / (self.key.out_features ** 0.5), dim=-1)
        output = attn_weights @ v

        return output, attn_weights


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

    def __init__(self, args):
        super(RNDNetwork, self).__init__()
        self.INPUT_DIM = args.INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = args.HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = args.OUTPUT_DIM  # 输出层的大小
        self.HIDDEN_DIM_NUM = args.HIDDEN_DIM_NUM
        self.LR = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor = Net(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.HIDDEN_DIM_NUM).to(self.device)
        init.normal_(self.predictor.input_layer.weight, mean=0, std=0.01)
        for layer in self.predictor.hidden_layers:
            init.normal_(layer.weight, mean=0, std=0.01)
        init.normal_(self.predictor.output_layer.weight, mean=0, std=0.01)
        self.target = Net(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.HIDDEN_DIM_NUM).to(self.device)
        init.normal_(self.target.input_layer.weight, mean=0, std=0.01)
        for layer in self.target.hidden_layers:
            init.normal_(layer.weight, mean=0, std=0.01)
        init.normal_(self.target.output_layer.weight, mean=0, std=0.01)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

        self.sum_error = 0.0
        self.running_std_error = 0.0
        self.running_mean_deviation = 0.0
        self.data_count = 0.0

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

    def get_intrinsic_reward(self, predict, target, CALC=True):
        """
        CALC参数指定是否将误差用于计算运行时标准差和均差
        """
        intrinsic_reward = self.update_parameters(predict, target)
        if CALC:
            self.get_runtime_std_mean_deviation(intrinsic_reward.item())

        return intrinsic_reward

    def get_runtime_std_mean_deviation(self, data):  # 计算运行时的标准差与均差
        self.data_count += 1
        delta = data - np.mean(self.sum_error)
        self.running_std_error += ((delta ** 2 - self.running_std_error) / self.data_count)
        self.running_mean_deviation += (np.abs(delta) / self.data_count)

        self.sum_error += data


class RNDNetwork_CNN(nn.Module):
    """
    该类基于随机网络蒸馏(Random Network Distillation, RND)的思想，基于预测误差给予智能体一定的内在奖励。
    通常用于鼓励智能体更多的探索新的状态。
    """

    def __init__(self, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM):
        super(RNDNetwork_CNN, self).__init__()
        self.INPUT_DIM = INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = OUTPUT_DIM  # 输出层的大小

        self.LR = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.target = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)

        self.recorder = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.recorder_target = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.initial_record = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)

        for param in self.target.parameters():
            param.requires_grad = False
        for param in self.recorder_target.parameters():
            param.requires_grad = False
        for param in self.initial_record.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.LR)
        self.recorder_optimizer = torch.optim.Adam(self.recorder.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

        self.sum_error = 0.0
        self.running_std_error = 0.0
        self.running_mean_deviation = 0.0
        self.data_count = 0.0

        self.mean = 0.0
        self.var = 1.0

    def forward(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        predict = self.predictor(state)
        target = self.target(state)
        return predict, target

    def update_parameters(self, predict, target):
        loss = self.loss_func(predict, target)
        loss_item = loss.item()
        normalized_error = (loss_item - self.mean) / math.sqrt(self.var)
        self.mean = 0.99 * self.mean + 0.01 * loss_item
        self.var = 0.99 * self.var + 0.01 * (loss_item ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.predictor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return normalized_error

    def get_intrinsic_reward(self, predict, target, CALC=True):
        """
        CALC参数指定是否将误差用于计算运行时标准差和均差
        """
        intrinsic_reward = self.update_parameters(predict, target)

        return intrinsic_reward

    def get_runtime_std_mean_deviation(self, data):  # 计算运行时的标准差与均差
        """
        该函数已废弃不用
        :param data:
        :return:
        """
        self.data_count += 1
        delta = data - np.mean(self.sum_error)
        self.running_std_error += ((delta ** 2 - self.running_std_error) / self.data_count)
        self.running_mean_deviation += (np.abs(delta) / self.data_count)

        self.sum_error += data


class RRNA(nn.Module):   # Reduce repeat nonsense action
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(MyNet, self).__init__()
        self.recorder = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_DIM)
        )
        self.target = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_DIM)
        )
        self.initial = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_DIM)
        )
        self.initial.load_state_dict(self.recorder.state_dict())
        self.LR = 1e-5
        self.optimizer = torch.optim.Adam(self.recorder.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    def forward(self, state_action):
        record = self.recorder(state_action)
        target = self.target(state_action)

        return record, target

    def learn(self, state_action, record, target):
        initial_output = self.initial(state_action).norm()  # 网络的初始误差
        current_error = self.loss_func(record, target)  # 当前的误差
        self.optimizer.zero_grad()
        current_error.backward()
        self.optimizer.step()

        # initial_error = self.loss_func(initial_output, record)
        normalize_error = current_error / (initial_output)

        return normalize_error.item()

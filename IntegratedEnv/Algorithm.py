# -*- coding = utf-8 -*-
# @time: 5/26/2023 9:17 PM
# Author: Yu Xia
# @File: Algorithm.py
# @software: PyCharm
import copy
import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from Network import *
from Tools import *
from replaybuffer import *
from collections import deque

from labml import tracker, experiment, logger, monit
from labml.internal.configs.dynamic_hyperparam import FloatDynamicHyperParam
from labml_helpers.schedule import Piecewise
from labml_nn.rl.dqn import QFuncLoss
from dqn_model import Model
from labml_nn.rl.dqn.replay_buffer import ReplayBuffer
from Wrapper import Worker
from Tools import Painter
from datetime import datetime
import os


class DQN:
    def __init__(self, args: SetupArgs, *, NAME="DQN"):
        self.NAME = NAME
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.INPUT_DIM = args.INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = args.HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = args.OUTPUT_DIM  # 输出层的大小
        self.HIDDEN_DIM_NUM = args.HIDDEN_DIM_NUM  # 隐藏层的数量
        self.SIZEOF_EVERY_MEMORY = args.SIZEOF_EVERY_MEMORY  # 经验回放的样本大小，使用不同环境时应进行修改。
        self.MEMORY_SHAPE = args.MEMORY_SHAPE  # 经验回放的样本中s, a, r, s_prime所占大小，使用不同环境时应进行修改。

        self.TARGET_REPLACE_ITER = 100
        self.MEMORY_CAPACITY = 1000000
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
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        else:
            state = state.unsqueeze(0).to(self.device)
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
            self.epsilon *= 0.9995

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
            batch_memory[:, self.MEMORY_SHAPE[0]: self.MEMORY_SHAPE[0] + self.MEMORY_SHAPE[1]].astype(int)).to(
            self.device)
        batch_r = torch.FloatTensor(batch_memory[:, self.MEMORY_SHAPE[0] + self.MEMORY_SHAPE[1]: self.MEMORY_SHAPE[0] +
                                                                                                 self.MEMORY_SHAPE[1] +
                                                                                                 self.MEMORY_SHAPE[
                                                                                                     2]]).to(
            self.device)
        batch_s_prime = torch.FloatTensor(
            batch_memory[:, self.MEMORY_SHAPE[0] + self.MEMORY_SHAPE[1] + self.MEMORY_SHAPE[2]: self.MEMORY_SHAPE[0] +
                                                                                                self.MEMORY_SHAPE[1] +
                                                                                                self.MEMORY_SHAPE[2] +
                                                                                                self.MEMORY_SHAPE[
                                                                                                    3]]).to(self.device)
        if len(self.MEMORY_SHAPE) == 5:
            batch_done = torch.LongTensor(batch_memory[:, -self.MEMORY_SHAPE[-1]:]).to(self.device)

        estimated_q = self.main_net(batch_s).gather(1, batch_a)
        if self.NAME == "DQN":
            q_target = self.target_net(batch_s_prime).detach()
        elif self.NAME == "DDQN":
            q_values_prime = self.main_net(batch_s_prime)
            best_actions = torch.argmax(q_values_prime, dim=1)
            q_target = self.target_net(batch_s_prime).detach()
            # y = batch_r + self.args.gamma * q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1) * (1 -
            # batch_done)

        if len(self.MEMORY_SHAPE) == 5:
            y = batch_r + self.args.gamma * q_target.max(1)[0].view(self.BATCH_SIZE, 1) * (1 - batch_done)
        else:
            y = batch_r + self.args.gamma * q_target.max(1)[0].view(self.BATCH_SIZE, 1)

        loss = self.loss_func(estimated_q, y)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


class DQN_CNN:
    def __init__(self, args: SetupArgs, *, NAME="DQN"):
        self.NAME = NAME
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.INPUT_DIM = args.INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = args.HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = args.OUTPUT_DIM  # 输出层的大小
        self.HIDDEN_DIM_NUM = args.HIDDEN_DIM_NUM  # 隐藏层的数量

        self.TARGET_REPLACE_ITER = 250
        self.LR = self.args.lr
        self.BATCH_SIZE = 32

        self.epsilon = self.args.epsilon
        self.learn_step_counter = 0

        self.memory_size = 2 ** 16
        self.memory = ReplayBuffer(self.memory_size, 4)
        self.memory_counter = 0
        self.learn_frequency = 0  # 记录执行了多少次step方法，控制经验回放的速率
        self.frame_count = 0  # 记录更新过多少帧

        self.main_net = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.target_net = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.LR)  # 网络参数优化
        self.loss_func = nn.MSELoss()  # 损失函数，默认使用均方误差损失函数

    @torch.no_grad()
    def select_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.OUTPUT_DIM)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            action = self.main_net(state).argmax().item()

        return action

    def memory_reset(self):
        self.memory = ReplayBuffer(self.memory_size, 4)
        self.memory_counter = 0
        self.learn_frequency = 0

    def step(self, index, a, r, done) -> float:
        """
        得到下一个观测之后才能调用
        """
        # if self.epsilon > 0.01:
        #     self.epsilon *= 0.9995

        self.memory.store_memory_effect(index, a, r, done)
        self.learn_frequency += 1
        if self.memory.memory_counter > self.memory.learning_starts and self.learn_frequency % 4 == 0:
            loss = self.learn()
            return loss
        return 0.0

    def change_to_tensor(self, data, dtype=torch.float32):
        """
        change ndarray to torch.tensor
        """
        data_tensor = torch.from_numpy(data).type(dtype)
        return data_tensor

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        self.learn_step_counter += 1
        batch_s, batch_a, batch_r, batch_s_prime, batch_done = self.memory.sample_memories(self.BATCH_SIZE)
        batch_s, batch_s_prime = self.change_to_tensor(batch_s).to(self.device) / 255.0, self.change_to_tensor(
            batch_s_prime).to(self.device) / 255.0
        batch_a, batch_r = self.change_to_tensor(batch_a, dtype=torch.int64).to(self.device), self.change_to_tensor(
            batch_r).to(self.device)
        batch_done = self.change_to_tensor(batch_done, dtype=torch.int64).to(self.device)

        self.frame_count += self.BATCH_SIZE

        estimated_q = self.main_net(batch_s)
        estimated_q = estimated_q.gather(1, batch_a)
        if self.NAME == "DQN":
            q_target = self.target_net(batch_s_prime).detach()
        elif self.NAME == "DDQN":
            q_values_prime = self.main_net(batch_s_prime)
            best_actions = torch.argmax(q_values_prime, dim=1)
            q_target = self.target_net(batch_s_prime).detach()

        y = batch_r + self.args.gamma * q_target.max(1)[0].view(self.BATCH_SIZE, 1) * ((1 - batch_done).unsqueeze(1))
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
        self.epsilon = args.epsilon

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
        if self.epsilon > 0.01:
            self.epsilon *= 0.99
        return self.update(s, a, r, s_prime, done)


class NGU:
    def __init__(self, K):
        self.episode_states = []
        self.episode_states_count = 0
        self.K = K
        self.k_nearest_neighbor = []
        self.L = 5
        self.c = 0.001
        self.epsilon = 0.001

        self.mean_value_of_k_neighbor = 0.0

    def store_state(self, state):
        self.episode_states.append(state)
        self.episode_states_count += 1

    def get_k_nearest_neighbor(self, new_state):
        if self.episode_states_count < self.K:
            return

        distances = [np.linalg.norm(new_state - state) for state in self.episode_states]
        k_nearest_indices = np.argsort(distances)[:self.K]
        self.k_nearest_neighbor = [self.episode_states[i] for i in k_nearest_indices]
        self.mean_value_of_k_neighbor = np.mean([np.sum(np.square(neighbor)) for neighbor in self.k_nearest_neighbor])

    def RBF_kernel_function(self, x, y):
        condition_number = np.linalg.norm(x - y)
        return self.epsilon / ((condition_number / self.mean_value_of_k_neighbor) + self.epsilon)

    def get_intrinsic_reward(self, new_state):
        if self.episode_states_count < self.K * 3:
            return 0.0
        else:
            self.get_k_nearest_neighbor(new_state)

            calc = 0.0
            for neighbor in self.k_nearest_neighbor:
                calc += self.RBF_kernel_function(new_state, neighbor)
            calc = np.sqrt(calc)
            return 1.0 / (calc + self.c)

    def reset(self):
        self.episode_states = []
        self.episode_states_count = 0
        self.mean_value_of_k_neighbor = 0.0
        self.k_nearest_neighbor = []


"""
----- ----------------------------------以下是做实验的代码----------------------------------
"""


class SuperBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, target):
        target_q = target.detach().cpu()  # target是网络输出，需要额外脱离gpu以及计算图
        current_state = state.cpu()  # 这里的state是tensor
        self.buffer.append((current_state, action, target_q))

    def sample(self, batch_size):
        # transitions = random.sample(self.buffer, batch_size)
        temp = list(self.buffer)
        transitions = temp[-batch_size:]

        state, action, target = transitions[0]
        return state, action, target

    def change(self, fact):
        # 动态调整buff大小
        temp = self.bufferModule
        self.buffer = deque(maxlen=self.capacity * fact)
        while len(temp) > 0:
            self.buffer.append(temp.pop())

    def __len__(self):
        print(self.buffer)
        return len(self.buffer)


class Super_net:
    def __init__(self, model):
        self.model = model
        self.lr = 1e-5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.MSELoss()
        self.buffer = SuperBuffer(1)
        self.batch_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, state):
        return self.model(state)

    def learn(self):
        if len(self.buffer.buffer) < self.batch_size:
            return torch.tensor(0, dtype=torch.float)
        state, action, target = self.buffer.sample(self.batch_size)

        states = state.unsqueeze(0).to(self.device)
        actions = torch.tensor(action, dtype=torch.long, device=self.device)
        target = target.to(self.device)
        # target = torch.tensor(target, dtype=torch.float, device=self.device).view(-1, 1).squeeze(0)

        super_value = self.model(states).squeeze(0)[actions]
        # super_value.gather(1, actions)

        loss = self.loss_func(super_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss


class DQN_CNN_Super:
    def __init__(self, args: SetupArgs, *, NAME="DQN"):
        self.NAME = NAME
        self.args = args

        self.interview_count = 0  # 记录智能体与环境交互的次数
        self.super_train_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.INPUT_DIM = args.INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = args.HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = args.OUTPUT_DIM  # 输出层的大小
        self.HIDDEN_DIM_NUM = args.HIDDEN_DIM_NUM  # 隐藏层的数量

        self.TARGET_REPLACE_ITER = 10000
        self.LR = self.args.lr
        self.BATCH_SIZE = 32

        self.epsilon = self.args.epsilon
        self.steps_done = 0  # 记录epsilon的衰减次数，得到下一次选择动作时的epsilon值。
        self.decay_start = self.epsilon
        self.decay_end = 0.01
        self.decay_step = 1000000

        self.learn_step_counter = 0

        self.memory_size = self.args.buffer_size
        self.memory = ReplayBuffer(self.memory_size, 4)
        self.memory_counter = 0
        self.learn_frequency = 0  # 记录执行了多少次step方法，控制经验回放的速率
        self.frame_count = 0  # 记录更新过多少帧

        self.main_net = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.target_net = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.main_net_deepcopy = copy.deepcopy(self.main_net)
        model = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        model.load_state_dict(self.main_net_deepcopy.state_dict())
        self.super_net = Super_net(model)

        self.super_net_init = CNN(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.super_net_init.load_state_dict(model.state_dict())
        self.super_net_init.eval()

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.LR)  # 网络参数优化
        self.loss_func = nn.MSELoss()  # 损失函数，默认使用均方误差损失函数
        self.l1loss_func = nn.L1Loss()

    @torch.no_grad()
    def select_action(self, state):
        q_value = float('-inf')
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.OUTPUT_DIM)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_action = self.main_net(state)
            action = q_action.argmax().item()
            q_value = q_action[0][action]

        return q_value, action

    def memory_reset(self):
        self.memory = ReplayBuffer(self.memory_size, 4)
        self.memory_counter = 0
        self.learn_frequency = 0

    def get_super_reward(self, q_val, state, action, reward):
        self.interview_count += 1
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        if q_val == float('-inf'):
            q_value = self.main_net(state).squeeze(0)
            q_value = q_value[action]
        else:
            q_value = q_val
        super_value = self.super_net(state).squeeze()
        super_value = super_value[action]
        reward_finally = 0
        if super_value < q_value:
            self.super_train_count += 1
            self.super_net.buffer.add(state, action, q_value)
            self.super_net.learn()
            reward_loss = self.l1loss_func(super_value, q_value).item()
            reward_distance = self.super_net(state).squeeze(0)[action].item() - super_value.item()

            reward_finally = min(reward_loss, reward_distance)

        return reward_finally

    def step(self, index, s, a, r, done) -> float:
        """
        得到下一个观测之后才能调用
        """
        # if self.epsilon > 0.01:
        #     self.epsilon *= 0.9995

        self.memory.store_memory_effect(index, a, r, done)
        self.learn_frequency += 1
        # print("Memory len:{}".format(self.memory.memory_counter))
        if self.memory.memory_counter > self.BATCH_SIZE and self.memory.memory_counter > self.memory.learning_starts and self.learn_frequency % 5 == 0:
            loss = self.learn()
            return loss
        return 0.0

    def change_to_tensor(self, data, dtype=torch.float32):
        """
        change ndarray to torch.tensor
        """
        data_tensor = torch.from_numpy(data).type(dtype)
        return data_tensor

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        self.learn_step_counter += 1
        batch_s, batch_a, batch_r, batch_s_prime, batch_done = self.memory.sample_memories(self.BATCH_SIZE)
        batch_s, batch_s_prime = self.change_to_tensor(batch_s).to(self.device) / 255.0, self.change_to_tensor(
            batch_s_prime).to(self.device) / 255.0
        batch_a, batch_r = self.change_to_tensor(batch_a, dtype=torch.int64).to(self.device), self.change_to_tensor(
            batch_r).to(self.device)
        batch_done = self.change_to_tensor(batch_done, dtype=torch.int64).to(self.device)

        self.frame_count += self.BATCH_SIZE

        estimated_q = self.main_net(batch_s)
        estimated_q = estimated_q.gather(1, batch_a)
        if self.NAME == "DQN":
            q_target = self.target_net(batch_s_prime).detach()
        elif self.NAME == "DDQN":
            q_values_prime = self.main_net(batch_s_prime)
            best_actions = torch.argmax(q_values_prime, dim=1)
            q_target = self.target_net(batch_s_prime).detach()

        y = batch_r + self.args.gamma * q_target.max(1)[0].view(self.BATCH_SIZE, 1) * ((1 - batch_done).unsqueeze(1))
        loss = self.loss_func(estimated_q, y)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


"""
-----------------------------------------以下是采取了多线程训练的代码-------------------------------------------------
"""
device = "cuda" if torch.cuda.is_available() else "cpu"


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class DQN_Super_Trainer:
    def __init__(self, updates: int,
                 epochs: int,
                 n_workers: int,
                 worker_steps: int,
                 mini_batch_size: int,
                 update_target_model: int,
                 learning_rate: FloatDynamicHyperParam,
                 args: SetupArgs,
                 use_super: bool,
                 rnd: dict,
                 test: dict,
                 algorithm_name="DQN"):
        self.args = args
        self.INPUT_DIM = args.INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = args.HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = args.OUTPUT_DIM  # 输出层的大小
        self.HIDDEN_DIM_NUM = args.HIDDEN_DIM_NUM  # 隐藏层的数量

        self.test = test['use_test']    # 是否加载模型并进行测试
        self.test_model = test['test_model']

        self.use_super = use_super

        """----------RND网络参数定义部分----------"""
        self.use_rnd = rnd['use_rnd']
        self.rnd_weight = rnd['rnd_weight']
        self.rnd_weight_decay = rnd['rnd_weight_decay']
        """----------RND网络参数定义结束----------"""

        self.algorithm_name = algorithm_name

        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.train_epochs = epochs
        self.updates = updates
        self.mini_batch_size = mini_batch_size
        self.update_target_model_frequency = update_target_model
        self.learning_rate = learning_rate
        if not self.test:
            self.exploration_coefficient = Piecewise(
                [
                    (0, 1.0),
                    (25000, 0.1),
                    (self.updates / 2, 0.01)
                ], outside_value=0.01
            )
        else:
            self.n_workers = 1
            self.exploration_coefficient = lambda x: 0.00001
        self.prioritized_replay_beta = Piecewise(
            [
                (0, 0.4),
                (self.updates, 1)
            ], outside_value=1
        )
        self.replay_buffer = ReplayBuffer(2 ** 14, 0.6)
        self.main_net = Model(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, algorithm_name=algorithm_name).to(
            device)
        self.target_net = Model(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, algorithm_name=algorithm_name).to(
            device)

        """--------------------------------Super网络的定义部分--------------------------------"""
        if self.use_super:
            self.main_copy = Model(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, algorithm_name=algorithm_name).to(
                device)
            self.main_copy.load_state_dict(self.main_net.state_dict())
            self.super_net = Super_net(self.main_copy)
            self.super_train_count = 0
        else:
            self.main_copy = None
            # self.main_copy.load_state_dict(self.main_net.state_dict())
            self.super_net = None
            self.super_train_count = 0
        """--------------------------------Super网络定义结束--------------------------------"""

        """--------------------------------RND网络的定义部分--------------------------------"""
        if self.use_rnd:
            self.RND_Network = RNDNetwork_CNN(args)
        else:
            self.RND_Network = None
        """--------------------------------RND网络的定义结束--------------------------------"""

        if self.test:
            checkpoint = torch.load(self.test_model)
            self.main_net.load_state_dict(checkpoint['main_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

        self.workers = [Worker(args, 47 + i, i) for i in range(self.n_workers)]
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)

        for worker in self.workers:
            worker.child.send(("reset", None))

        for i, worker in enumerate(self.workers):
            recv, info = worker.child.recv()
            self.obs[i] = recv

        self.loss_func = QFuncLoss(0.99)  # discount factor = 0.99
        self.l1loss_func = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=2.5e-4)

        self.painter = Painter()
        self.returns = []
        self.all_returns = []  # 把所有线程中得到的结果进行保存
        self.watch_processing = 3  # 指定绘制第几个线程的输出结果
        for i in range(self.n_workers):
            self.returns.append([])

    @torch.no_grad()
    def select_action(self, q_value, exploration_coefficient: float):
        greedy_action = torch.argmax(q_value, dim=-1)
        random_action = torch.randint(q_value.shape[-1], greedy_action.shape, device=q_value.device)
        is_choose_rand = torch.rand(greedy_action.shape, device=q_value.device) < exploration_coefficient
        return torch.where(is_choose_rand, random_action, greedy_action).cpu().numpy()

    def sample(self, exploration_coefficient: float):
        for t in range(self.worker_steps):
            state = obs_to_torch(self.obs)
            q_value = self.main_net(state)
            actions = self.select_action(q_value, exploration_coefficient)

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            for w, worker in enumerate(self.workers):
                s_prime, reward, done, info, _ = worker.child.recv()
                if self.use_super:
                    intrinsic_reward = self.get_super_reward(state[w], actions[w])
                    update_reward = reward + 0.8 * intrinsic_reward
                else:
                    update_reward = reward
                if self.use_rnd:
                    predict, target = self.RND_Network(state[w])
                    i_reward = self.RND_Network.get_intrinsic_reward(predict, target)
                    update_reward = update_reward + self.rnd_weight * i_reward
                self.replay_buffer.add(self.obs[w], actions[w], update_reward, s_prime, done)

                if info:
                    tracker.add('reward', info['reward'])
                    tracker.add('length', info['length'])
                    self.all_returns.append(info['reward'])
                    self.returns[w].append(info['reward'])
                    # if w == self.watch_processing and len(self.returns[w]) % 50 == 0:
                    # self.painter.plot_average_reward_by_list(self.returns[w][-50:], window=1, title="{} on {
                    # }".format("DQN", self.args.env_name), curve_label="{}".format("DQN" + " Super" if
                    # self.use_super else ""), colorIndex=self.watch_processing )

                self.obs[w] = s_prime

    def get_super_reward(self, state, action):
        q_values = self.main_net(state).squeeze(0)
        q_value = q_values[action]
        super_value = self.super_net(state).squeeze(0)
        super_value = super_value[action]
        reward_finally = 0
        if super_value < q_value:
            self.super_train_count += 1
            self.super_net.buffer.add(state, action, q_value)
            self.super_net.learn()
            reward_loss = self.l1loss_func(super_value, q_value).item()
            reward_distance = self.super_net(state).squeeze(0)[action].item() - super_value.item()

            reward_finally = min(reward_loss, reward_distance)

        return reward_finally

    def learn(self, beta: float):
        for _ in range(self.train_epochs):
            samples = self.replay_buffer.sample(self.mini_batch_size, beta)
            q_value = self.main_net(obs_to_torch(samples['obs']))

            with torch.no_grad():
                double_q_value = self.main_net(obs_to_torch(samples['next_obs']))
                target_q_value = self.target_net(obs_to_torch(samples['next_obs']))

            td_error, loss = self.loss_func(q_value,
                                            q_value.new_tensor(samples['action']),
                                            double_q_value, target_q_value,
                                            q_value.new_tensor(samples['done']),
                                            q_value.new_tensor(samples['reward']),
                                            q_value.new_tensor(samples['weights']))

            new_priorities = np.abs(td_error.cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(samples['indexes'], new_priorities)

            for pg in self.optimizer.param_groups:
                pg['lr'] = self.learning_rate()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=0.5)
            self.optimizer.step()

            return loss.item()

    def run_training_loop(self):
        tracker.set_queue('reward', 100, True)  # 跟踪显示100回合的平均奖励
        tracker.set_queue('length', 100, True)  # 跟踪显示100回合的平均回合长度

        self.target_net.load_state_dict(self.main_net.state_dict())
        for update in monit.loop(self.updates):
            exploration = self.exploration_coefficient(update)
            tracker.add('exploration', exploration)

            beta = self.prioritized_replay_beta(update)
            tracker.add('beta', beta)

            self.sample(exploration)
            if self.replay_buffer.is_full():
                self.learn(beta)

                if update % self.update_target_model_frequency == 0:
                    self.target_net.load_state_dict(self.main_net.state_dict())

            tracker.save()
            if (update + 1) % 1000 == 0:
                logger.log()
            if (update + 1) % 100000 == 0 and not self.test:
                message = "{}_rounds".format(update + 1) + "_super" if self.use_super else ""
                self.save_info(message=message, rounds=update)
            if (update + 1) % 600000 == 0:
                break
        if not self.test:
            message = "final" + "_super" if self.use_super else ""
            self.save_info(message=message, rounds=self.updates)

    def save_info(self, message="", rounds=0):
        formatted_time = datetime.now().strftime("%y_%m_%d_%H")
        name = self.args.env_name.split("/")[-1]
        torch.save({"main_net_state_dict": self.main_net.state_dict(),
                    "target_net_state_dict": self.target_net.state_dict()},
                   "./checkpoint/dqn_{}_{}_{}.pth".format(name, formatted_time, message))
        self.painter.plot_average_reward_by_list(None,
                                                 window=1,
                                                 title="{} on {}".format("DQN" + " Super" if self.use_super else "",
                                                                         self.args.env_name),
                                                 curve_label="{}".format("DQN" + " Super" if self.use_super else ""),
                                                 colorIndex=self.watch_processing,
                                                 savePath="./train_pic/dqn_{}_{}_{}.png".format(name, message,
                                                                                                formatted_time),
                                                 end=True
                                                 )
        for i in range(self.n_workers):
            fileName = './data/Process_{}_{}_{}.txt'.format(i, formatted_time, message)
            with open(fileName, 'w') as file_object:
                file_object.write(str(self.returns[i]))

        fileName = './data/All Process_{}_{}.txt'.format(message, formatted_time)
        with open(fileName, 'w') as file_object:
            file_object.write(str(self.all_returns))

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))

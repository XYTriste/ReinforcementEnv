# -*- coding = utf-8 -*-
# @time: 5/21/2023 12:14 PM
# Author: Yu Xia
# @File: MountainCarContinuous.py
# @software: PyCharm
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DQNAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  # 如果有GPU和cuda
        # ，数据将转移到GPU执行
        torch.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

        # 定义Actor网络和目标Actor网络
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        # 定义Critic网络和目标Critic网络
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()

        # 定义优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, batch_size=64, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
              policy_freq=2):
        # 从经验回放缓冲区中随机采样一个批次的经验
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # 计算目标动作
        next_action_batch = self.target_actor(next_state_batch)
        noise = torch.FloatTensor(action_batch).data.normal_(0, policy_noise).to(self.device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action_batch = (next_action_batch + noise).clamp(-self.actor.max_action, self.actor.max_action)

        # 计算目标Q值
        target_Q1 = self.target_critic1(next_state_batch, next_action_batch)
        target_Q2 = self.target_critic2(next_state_batch, next_action_batch)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward_batch + (1 - done_batch) * discount * target_Q

        # 更新Critic网络
        current_Q1 = self.critic1(state_batch, action_batch)
        current_Q2 = self.critic2(state_batch, action_batch)
        critic1_loss = F.mse_loss(current_Q1, target_Q.detach())
        critic2_loss = F.mse_loss(current_Q2, target_Q.detach())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 延迟更新Actor网络和目标网络
        if batch_size % policy_freq == 0:
            actor_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新目标网络的参数
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)

    def __len__(self):
        return len(self.buffer)


# 创建MountainCarContinuous环境
env = gym.make('MountainCarContinuous-v0', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# 创建DQN智能体
agent = DQNAgent(state_dim, action_dim, max_action)

# 定义训练参数
max_episodes = 1000
max_steps = 500
batch_size = 64
gamma = 0.99
tau = 0.005

# 创建经验回放缓冲区
replay_buffer = ReplayBuffer(2000)

# 开始训练
for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add(state, action, next_state, reward, float(done))
        state = next_state
        episode_reward += reward
        if done:
            break

    if len(replay_buffer) > batch_size:
        agent.train(replay_buffer, batch_size, gamma, tau)

    print("Episode: {}, Reward: {}".format(episode, episode_reward))

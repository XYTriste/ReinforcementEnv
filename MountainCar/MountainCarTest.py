# -*- coding = utf-8 -*-
# @time: 5/17/2023 10:32 AM
# Author: Yu Xia
# @File: MountainCarTest.py
# @software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  # 如果有GPU和cuda
# ，数据将转移到GPU执行
torch.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
# device = torch.device("cpu")


class Agent:
    def __init__(self, N_STATES, N_ACTIONS):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.STATES = None


class Net(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, OUTPUT_DIM)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions_value = self.out(state)

        return actions_value


class DQN:
    def __init__(self, agent: Agent):
        self.main_net, self.target_net = Net(2, 3).to(device), Net(2, 3).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.BATCH_SIZE = 32
        self.LR = 0.01
        self.EPSILON = 0.5
        self.GAMMA = 0.95
        self.TARGET_REPLACE_ITER = 100
        self.MEMORY_CAPACITY = 2000
        self.agent = agent
        self.EVERY_MEMORY_SIZE = agent.N_STATES * 2 + 2
        self.EVERY_MEMORY_SHAPE = (agent.N_STATES, 1, 1, agent.N_STATES)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.EVERY_MEMORY_SIZE))
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    @torch.no_grad()
    def epsilon_greedy_policy(self, state):
        if np.random.uniform(0, 1) < self.EPSILON:
            action = np.random.randint(0, self.agent.N_ACTIONS)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            actions_value = self.main_net(state)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]

        return action

    def store_transition(self, s, a, r, s_prime):
        transition = np.hstack((s, [a, r], s_prime))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, : self.EVERY_MEMORY_SHAPE[0]])
        batch_a = torch.LongTensor(
            batch_memory[:, self.EVERY_MEMORY_SHAPE[0]: self.EVERY_MEMORY_SHAPE[0] + 1].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, self.EVERY_MEMORY_SHAPE[0] + 1: self.EVERY_MEMORY_SHAPE[0] + 2])
        batch_s_prime = torch.FloatTensor(batch_memory[:, -self.EVERY_MEMORY_SHAPE[0]:])

        q = self.main_net(batch_s).gather(1, batch_a)
        q_target = self.target_net(batch_s_prime).detach()

        y = batch_r + self.GAMMA * q_target.max(1)[0].view(self.BATCH_SIZE, 1)

        loss = self.loss_func(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


if __name__ == '__main__':
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    DQNAgent = DQN(agent)

    flag = False
    rounds = 500
    for i in range(rounds):
        state, _ = env.reset()
        episode_reward = 0

        done = False
        while not done:
            action = DQNAgent.epsilon_greedy_policy(state)

            s_prime, r, done, _, _, = env.step(action)

            r = s_prime[0] + 0.5
            if s_prime[0] > -0.5:
                r = s_prime[0] + 0.5
                if s_prime[0] > 0.5:
                    r = 5
            else:
                r = 0

            DQNAgent.store_transition(state, action, r, s_prime)

            episode_reward += r
            if DQNAgent.memory_counter > DQNAgent.MEMORY_CAPACITY:
                DQNAgent.learn()
                if done:
                    print('Episode: ', i, '| Episode_reward: ', round(episode_reward, 2))

            state = s_prime

        if not flag and i > 50:
            env = gym.make("MountainCar-v0", render_mode="human")
            flag = True
        elif flag:
            env.render()

        if DQNAgent.EPSILON > 0.05:
            DQNAgent.EPSILON *= 0.99

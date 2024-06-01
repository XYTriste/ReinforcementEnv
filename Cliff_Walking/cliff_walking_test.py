import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
from labml import logger, experiment
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  # 如果有GPU和cuda
# ，数据将转移到GPU执行
torch.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class Agent:
    def __init__(self, N_STATES, N_ACTIONS):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.STATES = None


class Net(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_DIM)
        )

    def forward(self, state):
        actions_value = self.network(state)

        return actions_value


class DQN:
    def __init__(self, agent: Agent):
        self.main_net, self.target_net = Net(1, 3).to(device), Net(1, 3).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.BATCH_SIZE = 64
        self.LR = 1e-3
        self.EPSILON = 1
        self.GAMMA = 0.99
        self.TARGET_REPLACE_ITER = 100
        self.MEMORY_CAPACITY = 10000
        self.agent = agent
        self.EVERY_MEMORY_SIZE = agent.N_STATES * 2 + 2
        self.EVERY_MEMORY_SHAPE = (agent.N_STATES, 1, 1, agent.N_STATES)
        self.Q_targets = 0.0

        self.learn_step_counter = 0
        self.memory_counter = 0
        # self.memory = deque(maxlen=self.MEMORY_CAPACITY)
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.agent.N_STATES * 2 + 3))
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()
        self.huber_loss_func = nn.SmoothL1Loss()

    @torch.no_grad()
    def epsilon_greedy_policy(self, state):
        if np.random.uniform(0, 1) < self.EPSILON:
            action = np.random.randint(0, self.agent.N_ACTIONS)
        else:
            state = torch.tensor(state).unsqueeze(0).to(device).to(torch.float)
            # actions_value = self.main_net(state)
            # action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            # action = action[0]
            action = self.main_net(state).argmax().item()

        return action

    @torch.no_grad()
    def greedy_policy(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.main_net(state).argmax().item()
        # action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        # action = action[0]
        return action

    def store_transition(self, s, a, r, s_prime, done):
        transition = np.hstack((s, [a, r], s_prime, [done]))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        # e = self.experience(s, a, r, s_prime, done)
        # self.memory.append(e)
        # self.memory_counter += 1

    def learn(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 0
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, : self.EVERY_MEMORY_SHAPE[0]])
        batch_a = torch.LongTensor(
            batch_memory[:, self.EVERY_MEMORY_SHAPE[0]: self.EVERY_MEMORY_SHAPE[0] + 1].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, self.EVERY_MEMORY_SHAPE[0] + 1: self.EVERY_MEMORY_SHAPE[0] + 2])
        batch_s_prime = torch.FloatTensor(
            batch_memory[:, self.EVERY_MEMORY_SHAPE[0] + 2: self.EVERY_MEMORY_SHAPE[0] + 3])
        batch_done = torch.LongTensor(batch_memory[:, -1:])

        # samples = random.sample(self.memory, k=self.BATCH_SIZE)
        # batch_s = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        # batch_a = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        # batch_r = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(
        #     device)
        # batch_s_prime = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        # batch_done = torch.from_numpy(np.vstack([e.done for e in samples if e is not None])).long().to(
        #     device)

        q = self.main_net(batch_s).gather(1, batch_a)
        q_target = self.target_net(batch_s_prime).detach()

        y = batch_r + self.GAMMA * q_target.max(1)[0].view(self.BATCH_SIZE, 1) * (1 - batch_done)

        loss = self.huber_loss_func(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    agent = Agent(1, 3)
    DQNAgent = DQN(agent)
    oneDimension = []
    learn_step = 0

    invalid_action = 0
    invalid_dict = {}

    valid_action = 0
    rounds = 10000

    fig, ax = plt.subplots()
    invalid_state = []
    invalid_actions = []

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir_name = 'log/{}'.format(current_time)
    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)
    else:
        raise RuntimeError('log directory is exists.')

    writer = SummaryWriter(log_dir=log_dir_name)

    for i in range(10):
        with tqdm(total=int(rounds / 10), desc=f'Iteration {i}') as pbar:
            for episode in range(rounds // 10):
                state, _ = env.reset()

                done = False
                time_step = 0

                episode_DQN_loss = 0
                episode_learn_count = 0  # 每回合DQN算法学习的次数
                episode_reward = 0

                while not done:
                    action = DQNAgent.epsilon_greedy_policy(state)

                    time_step += 1

                    s_prime, extrinsic_reward, done, _, _, = env.step(action)

                    episode_reward += extrinsic_reward

                    euclidean_distance = np.linalg.norm(s_prime - state)
                    oneDimension.append(euclidean_distance)

                    DQNAgent.store_transition(state, action, extrinsic_reward, s_prime, done)
                    if s_prime == state:
                        invalid_action += 1
                        invalid_state.append(state)
                        invalid_actions.append(action)
                    else:
                        valid_action += 1

                    state = s_prime

                    learn_step += 1
                    if DQNAgent.memory_counter > DQNAgent.MEMORY_CAPACITY and learn_step % 5 == 0:
                        episode_learn_count += 1
                        episode_DQN_loss += DQNAgent.learn()

                if DQNAgent.EPSILON > 0.01:
                    DQNAgent.EPSILON *= 0.9999

                writer.add_scalar("Episode reward", episode_reward, i * (rounds // 10) + episode)
                writer.add_scalar("Episode DQN Loss", episode_DQN_loss, i * (rounds // 10) + episode)

                pbar.update(1)

    ax.plot(invalid_state, invalid_actions, 'o')
    ax.set_title('invalid action')

    file_name = f'result/invalid_actions_plot_{current_time}.png'
    plt.savefig(file_name)
    # print('invalid state:{}   action:{}'.format(state, action))
    writer.add_figure('invalid action', fig, global_step=None, close=False, walltime=None)

    print('valid action:', valid_action, ' invalid action:', invalid_action)
    for key, value in invalid_dict.items():
        print('invalid state: {},  action: {}'.format(key, value))

    with open('data/invalid action position.txt', 'w') as file_obj:
        for i in range(len(invalid_state)):
            line = "{},{}\n".format(invalid_state[i], invalid_actions[i])
            file_obj.write(line)

    # plt.show()
    # with open('data/oneDimension.txt', 'w') as file_object:
    #     for item in oneDimension:
    #         var = "{:12f}".format(item)
    #         line = var + "\n"
    #         file_object.write(line)

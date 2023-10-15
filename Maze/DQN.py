import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from MazeEnv import MazeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor  # 如果有GPU和cuda
# ，数据将转移到GPU执行
torch.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


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
    def __init__(self, agent: MazeEnv):
        self.main_net, self.target_net = Net(agent.N_STATES, agent.N_ACTIONS).to(device), Net(agent.N_STATES, agent.N_ACTIONS).to(device)
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
            state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)
            # actions_value = self.main_net(state)
            # action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            # action = action[0]
            action = self.main_net(state).argmax().item()

        return action

    @torch.no_grad()
    def greedy_policy(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.main_net(state).argmax().item()
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
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
        batch_s = torch.FloatTensor(batch_memory[:, : self.EVERY_MEMORY_SHAPE[0]]).to(device)
        batch_a = torch.LongTensor(
            batch_memory[:, self.EVERY_MEMORY_SHAPE[0]: self.EVERY_MEMORY_SHAPE[0] + 1].astype(int)).to(device)
        batch_r = torch.FloatTensor(batch_memory[:, self.EVERY_MEMORY_SHAPE[0] + 1: self.EVERY_MEMORY_SHAPE[0] + 2]).to(device)
        batch_s_prime = torch.FloatTensor(
            batch_memory[:, self.EVERY_MEMORY_SHAPE[0] + 2: self.EVERY_MEMORY_SHAPE[0] + 4]).to(device)
        batch_done = torch.LongTensor(batch_memory[:, -1:]).to(device)

        # samples = random.sample(self.memory, k=self.BATCH_SIZE)
        # batch_s = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        # batch_a = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        # batch_r = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(
        #     device)
        # batch_s_prime = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        # batch_done = torch.from_numpy(np.vstack([e.done for e in samples if e is not None])).long().to(
        #     device)
        q = self.main_net(batch_s).gather(1,batch_a)
        q_target = self.target_net(batch_s_prime).detach()

        y = batch_r + self.GAMMA * q_target.max(1)[0].view(self.BATCH_SIZE, 1) * (1 - batch_done)

        loss = self.huber_loss_func(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

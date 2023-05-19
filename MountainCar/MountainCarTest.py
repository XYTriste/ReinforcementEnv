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
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        # self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        # self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, OUTPUT_DIM)
        # self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions_value = self.out(state)

        return actions_value


class RNDNet(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(RNDNet, self).__init__()
        self.predictor = nn.Sequential(
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

        self.LR = 0.01
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

        self.normalize_min_error = float("inf")
        self.normalize_max_error = float("-inf")

    def forward(self, state):
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

    def normalize_error(self, predict_error):
        if predict_error < self.normalize_min_error:
            self.normalize_min_error = predict_error
        if predict_error > self.normalize_max_error:
            self.normalize_max_error = predict_error
        if self.normalize_min_error == self.normalize_max_error:
            self.normalize_min_error = self.normalize_max_error - 1E-3
        normalize_val = (predict_error - self.normalize_min_error) / (
                    self.normalize_max_error - self.normalize_min_error)
        normalize_val = normalize_val * 0.1 + 0.1

        return normalize_val


class DQN:
    def __init__(self, agent: Agent):
        self.main_net, self.target_net = Net(2, 3).to(device), Net(2, 3).to(device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.BATCH_SIZE = 32
        self.LR = 0.01
        self.EPSILON = 0.9
        self.GAMMA = 0.99
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

    @torch.no_grad()
    def greedy_policy(self, state):
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

        return loss.item()


class Recorder:
    def __init__(self):
        self.episodes = []
        self.time_steps = []
        self.episode_rewards = []
        self.predict_errors = []
        self.DQNLoss = []


class Painter:
    def __init__(self, x_values=[], y_values=[]):
        self.x_values = x_values
        self.y_values = y_values
        self.fig, self.ax = plt.subplots()

    def add_data(self, x, y):
        self.x_values.append(x)
        self.y_values.append(y)
        self.update_plot()

    def update_plot(self):
        # self.ax.clf()
        self.ax.plot(self.x_values, self.y_values)
        plt.show()


def run_evaluate_episodes(agent: DQN, eval_episodes=5, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        time_step = 0
        while not done and time_step < 200:
            time_step += 1
            action = agent.greedy_policy(state)
            s_prime, external_reward, done, _, _, = env.step(action)
            episode_reward += external_reward
            state = s_prime

        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


if __name__ == '__main__':
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    DQNAgent = DQN(agent)
    RNDNetWork = RNDNet(2, 2)
    recorder = Recorder()
    painter = Painter()

    flag = False
    rounds = 20
    data_iterator = tqdm(range(rounds), colour="green")

    for i in data_iterator:
        state, _ = env.reset()
        episode_reward = 0
        time_step = 0
        episode_external_reward = 0
        episode_intrinsic_reward = 0
        episode_predict_error = 0
        episode_DQN_loss = 0

        done = False
        while not done:
            action = DQNAgent.epsilon_greedy_policy(state)

            time_step += 1

            s_prime, external_reward, done, _, _, = env.step(action)

            external_reward = (s_prime[0] - state[0]) + (s_prime[1] ** 2 - state[1] ** 2)

            # if s_prime[0] > -0.5:
            #     external_reward = s_prime[0] + 0.5
            #     if s_prime[0] > 0.5:
            #         external_reward = 5
            # else:
            #     external_reward = 0

            # external_reward = 100 * (0.5 - abs(state[0])) - 10 * abs(state[1])
            # predict, target = RNDNetWork(torch.from_numpy(state))
            # # predict_error = np.linalg.norm(target - predict)
            # predict_error = RNDNetWork.update_parameters(predict, target)
            # normalize_val = RNDNetWork.normalize_error(predict_error.item())
            # episode_predict_error += normalize_val

            r = external_reward  # + normalize_val
            DQNAgent.store_transition(state, action, r, s_prime)

            episode_reward += r
            episode_external_reward += external_reward
            # episode_intrinsic_reward += normalize_val

            if DQNAgent.memory_counter > DQNAgent.MEMORY_CAPACITY:
                episode_DQN_loss += DQNAgent.learn()

                if done:
                    recorder.episodes.append(i)
                    recorder.time_steps.append(time_step)
                    recorder.episode_rewards.append(episode_reward)
                    recorder.predict_errors.append(episode_predict_error)
                    recorder.DQNLoss.append(episode_DQN_loss)
                    # painter.add_data(time_step, episode_reward)
                    # print('Episode: ', i,'| Episode_reward: ', round(episode_reward, 2))
                    print("Episode: {}, Episode reward:{}, external reward:{},   intrinsic reward:{}".format(i,
                                                                                                             episode_reward,
                                                                                                             episode_external_reward,
                                                                                                             episode_intrinsic_reward))
            state = s_prime

        if DQNAgent.EPSILON > 0.01:
            DQNAgent.EPSILON *= 0.99
        if i > 0 and i % 50 == 0:
            mean_reward = run_evaluate_episodes(DQNAgent)
            print("Episode:{}   epsilon:{}   Mean reward:{}".format(i, DQNAgent.EPSILON, mean_reward))

        # if not flag and i > 50:
        #     env = gym.make("MountainCar-v0", render_mode="human")
        #     flag = True
        # elif flag:
        #     env.render()

    data_iterator.close()
    plt.plot(recorder.episodes, recorder.DQNLoss, label="DQN Loss")
    plt.plot(recorder.episodes, recorder.episode_rewards, label="Episode rewards")
    # plt.plot(recorder.episodes, recorder.predict_errors, label="predict_error")
    plt.xlabel('Episodes')
    plt.show()

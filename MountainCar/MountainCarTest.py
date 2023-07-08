# -*- coding = utf-8 -*-
# @time: 5/17/2023 10:32 AM
# Author: Yu Xia
# @File: MountainCarTest.py
# @software: PyCharm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
from collections import deque

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


class RNDNet(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(RNDNet, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUT_DIM)
        )

        self.target = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUT_DIM)
        )

        self.LR = 0.001
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
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
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
            batch_memory[:, self.EVERY_MEMORY_SHAPE[0] + 2: self.EVERY_MEMORY_SHAPE[0] + 4])
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

    def plot_reward(self, reward_list, window, end=False):
        plt.figure(window)
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title("DQN on MountainCar-v0")
        list_t = [np.mean(reward_list[:i]) for i in range(len(reward_list))]
        plt.plot(np.array(list_t))
        if end:
            plt.show()
        else:
            plt.pause(0.001)


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
    RNDNetWork = RNDNet(2, 1)
    recorder = Recorder()
    painter = Painter()

    flag = False
    rounds = 1000

    test_rounds = []
    test_mean_reward = []
    learn_step = 0  # DQN经验回放频率计数

    reward_weight = 0.01

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)

    for i in range(10):
        with tqdm(total=int(rounds / 10), desc=f'Iteration {i}') as pbar:
            for episode in range(rounds // 10):
                state, _ = env.reset(seed=21)
                episode_reward = 0
                time_step = 0

                episode_external_reward = 0
                episode_intrinsic_reward = 0
                episode_predict_error = 0
                episode_DQN_loss = 0
                episode_learn_count = 0  # 每回合DQN算法学习的次数

                done = False
                while not done and time_step < 200:
                    action = DQNAgent.epsilon_greedy_policy(state)

                    time_step += 1

                    s_prime, extrinsic_reward, done, _, _, = env.step(action)

                    # external_reward = (s_prime[0] - state[0]) + (s_prime[1] ** 2 - state[1] ** 2)

                    # if s_prime[0] > -0.5:
                    #     external_reward = s_prime[0] + 0.5
                    #     if s_prime[0] > 0.5:
                    #         external_reward = 5
                    # else:
                    #     external_reward = 0

                    # external_reward = 100 * (0.5 - abs(state[0])) - 10 * abs(state[1])
                    predict, target = RNDNetWork(torch.from_numpy(state))
                    # # predict_error = np.linalg.norm(target - predict)
                    # predict_error = RNDNetWork.update_parameters(predict, target)
                    # normalize_val = RNDNetWork.normalize_error(predict_error.item())
                    # episode_predict_error += normalize_val
                    #
                    # r = external_reward + normalize_val
                    # r = (-external_reward) * predict_error.item()
                    if done:
                        special_reward = 0
                    else:
                        special_reward = 0
                    r = extrinsic_reward + reward_weight * (target - predict).item() + special_reward
                    # r = external_reward if not done else 0
                    DQNAgent.store_transition(state, action, r, s_prime, done)

                    episode_reward += extrinsic_reward
                    episode_external_reward += extrinsic_reward
                    episode_intrinsic_reward += reward_weight * (target - predict).item()
                    state = s_prime
                    learn_step += 1
                    if DQNAgent.memory_counter > DQNAgent.MEMORY_CAPACITY and learn_step % 5 == 0:
                        episode_learn_count += 1
                        episode_DQN_loss += DQNAgent.learn()

                    if DQNAgent.EPSILON > 0.01:
                        DQNAgent.EPSILON *= 0.9999
                recorder.episodes.append(i)
                recorder.time_steps.append(time_step)
                recorder.episode_rewards.append(episode_reward)
                recorder.predict_errors.append(episode_predict_error)
                recorder.DQNLoss.append(episode_DQN_loss / (episode_learn_count if episode_learn_count != 0 else 1))
                # painter.add_data(time_step, episode_reward)
                # print('Episode: ', i,'| Episode_reward: ', round(episode_reward, 2))
                print(
                    "Episode: {}, Episode reward:{}, extrinsic reward:{:.3f},   intrinsic reward:{:.3f}.   episode DQN "
                    "mean loss:{:.4f}, epsilon:{:.3f}".format(
                        i,
                        episode_reward,
                        episode_external_reward,
                        episode_intrinsic_reward,
                        episode_DQN_loss / (episode_learn_count if episode_learn_count != 0 else 1), DQNAgent.EPSILON))
                DQNAgent.plot_reward(recorder.episode_rewards, 1)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": f"{rounds / 10 * i + episode + 1}",
                            "return": f"{np.mean(recorder.episode_rewards[-10:]):3f}"
                        }
                    )
                pbar.update(1)

                if episode > 0 and episode % 50 == 0:
                    test_rounds.append(episode * (i + 1))
                    mean_reward = run_evaluate_episodes(DQNAgent)
                    test_mean_reward.append(mean_reward)
                    print("Episode:{}   epsilon:{}   Mean reward:{}".format(i, DQNAgent.EPSILON, mean_reward))

        # if not flag and i > 50:
        #     env = gym.make("MountainCar-v0", render_mode="human")
        #     flag = True
        # elif flag:
        #     env.render()

    # plt.plot(recorder.episodes, recorder.DQNLoss, label="DQN Loss")
    # plt.plot(recorder.episodes, recorder.episode_rewards, label="Episode rewards")
    # plt.plot(test_rounds, test_mean_reward, label="Average reward pre 5 rounds")
    # plt.legend()
    # plt.plot(recorder.episodes, recorder.predict_errors, label="predict_error")
    # plt.xlabel('Episodes')
    plt.show()

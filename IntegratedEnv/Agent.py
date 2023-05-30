# -*- coding = utf-8 -*-
# @time: 5/26/2023 10:15 PM
# Author: Yu Xia
# @File: Agent.py
# @software: PyCharm
import gymnasium as gym
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from Tools import Painter
from Network import *


class Agent:
    def __init__(self, args, env=None, algorithm=None):
        self.env = env
        self.algorithm = algorithm
        self.args = args
        self.painter = Painter()

    def train(self, use_rnd=True):
        RndNet = RNDNetwork()
        RND_WEIGHT = 0.4
        num_episodes = self.args.num_episodes
        return_list = []

        for i in range(10):
            Iteration_reward = []
            with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
                for episode in range(num_episodes // 10):
                    state, _ = self.env.reset(seed=self.args.seed)

                    time_step = 0
                    done = False

                    episode_reward = 0

                    while not done and time_step < 200:
                        action = self.algorithm.select_action(state)

                        s_prime, reward, done, _, _ = self.env.step(action)

                        # reward += (s_prime[0] - state[0]) + (s_prime[1] ** 2 - state[1] ** 2)

                        if use_rnd:
                            predict, target = RndNet(torch.from_numpy(state))
                            intrinsic_reward = RndNet.get_intrinsic_reward(predict, target).item()

                            update_reward = ((1 - RND_WEIGHT) * reward + RND_WEIGHT * intrinsic_reward)
                        else:
                            update_reward = reward

                        loss = self.algorithm.step(state, action, update_reward, s_prime, done)
                        #  算法每回合执行的一些步骤，里面包含更新网络等内容。返回网络的损失值

                        episode_reward += reward
                        state = s_prime

                        time_step += 1

                    return_list.append(episode_reward)
                    Iteration_reward.append(episode_reward)

                    # 预留位置给绘制图形的函数

                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{num_episodes / 10 * i + episode + 1}",
                                "return of last 10 rounds": f"{np.mean(return_list[-10:]):3f}",
                                "Iteration average reward:": f"{np.mean(Iteration_reward):3f}"
                            }
                        )
                    pbar.update(1)

        if use_rnd:
            self.painter.plot_average_reward(return_list, 1,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "DQN+RND",
                                             "red")
            plt.legend()
            self.painter.plot_episode_reward(return_list, 2,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "DQN+RND",
                                             "red")
        else:
            self.painter.plot_average_reward(return_list, 1,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "DQN",
                                             "blue")
            plt.legend()
            self.painter.plot_episode_reward(return_list, 2,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "DQN",
                                             "blue")
        plt.legend()

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
from Algorithm import NGU


class Agent:
    def __init__(self, args, env=None, algorithm=None):
        self.env = env
        self.algorithm = algorithm
        self.args = args
        self.painter = Painter()

    def train(self, use_rnd=False, rnd_weight_decay=1.0, use_ngu=False, painter_label=1):
        RndNet = RNDNetwork()
        RND_WEIGHT = 0.4

        ngu = NGU(3)

        num_episodes = self.args.num_episodes
        return_list = []
        average_loss = []

        player_win = 0
        dealer_win = 0
        rounds = 0

        min_reward = 1.0
        max_reward = -0.001

        for i in range(10):
            Iteration_reward = []
            with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
                for episode in range(num_episodes // 10):
                    state, _ = self.env.reset()

                    time_step = 0
                    done = False

                    episode_reward = 0
                    rounds += 1

                    ngu.store_state(state)

                    while not done and time_step < 200:
                        action = self.algorithm.select_action(state)

                        s_prime, reward, done, _, _ = self.env.step(action)

                        # reward += (s_prime[0] - state[0]) + (s_prime[1] ** 2 - state[1] ** 2)

                        if use_rnd:
                            predict, target = RndNet(torch.from_numpy(state))
                            intrinsic_reward = RndNet.get_intrinsic_reward(predict, target).item()

                            if use_ngu:
                                ngu_reward = ngu.get_intrinsic_reward(s_prime)
                                alpha_t = 1 + ((RndNet.sum_error - RndNet.running_mean_deviation) / RndNet.running_std_error)
                                ngu_reward = ngu_reward * min(max(alpha_t, 1), ngu.L)

                                normalized_reward = ((ngu_reward - min_reward) / (max_reward - min_reward)) * 0.25   # 归一化
                                min_reward = min(min_reward, max(ngu_reward, 0.001))
                                max_reward = max(ngu_reward, max_reward)

                                if ngu.episode_states_count > ngu.K * 2:
                                    intrinsic_reward += normalized_reward
                                    # print("normalize reward:", normalized_reward)

                            update_reward = ((1 - RND_WEIGHT) * reward + RND_WEIGHT * intrinsic_reward)
                        else:
                            update_reward = reward

                        # if update_reward >= 0:
                        #     print('it is inaccessible')
                        loss = self.algorithm.step(state, action, update_reward, s_prime, done)
                        #  算法每回合执行的一些步骤，里面包含更新网络等内容。返回网络的损失值

                        episode_reward += reward
                        state = s_prime
                        ngu.store_state(state)

                        time_step += 1
                        average_loss.append(loss)

                        if done:
                            if reward > 0:
                                player_win += 1
                            elif reward < 0:
                                dealer_win += 1
                            # print("Player state:{}   Dealer state:{}".format(s_prime[0], s_prime[1]))

                    return_list.append(episode_reward)
                    Iteration_reward.append(episode_reward)

                    if use_rnd:
                        RND_WEIGHT *= rnd_weight_decay
                        if use_ngu:
                            ngu.reset()

                    # 预留位置给绘制图形的函数

                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{num_episodes / 10 * i + episode + 1}",
                                "return of last 10 rounds": f"{np.mean(return_list[-10:]):3f}",
                                "Iteration average reward:": f"{np.mean(Iteration_reward):3f}",
                                "loss of last 10 rounds": f"{np.mean(average_loss[-10:]):9f}",
                                # "Player win rate": f"{(player_win / rounds):3f}",
                                # "Dealer win rate": f"{(dealer_win / rounds):3f}"
                            }
                        )
                    pbar.update(1)

        # if use_rnd:
        #     self.painter.plot_average_reward(return_list, 1,
        #                                      "{} on {}".format(self.algorithm.NAME, self.args.env_name),
        #                                      "{} + RND + NGU".format(self.algorithm.NAME),
        #                                      "red" if painter_label == 1 else "blue")
        #     plt.legend()
        #     self.painter.plot_episode_reward(return_list, 2,
        #                                      "{} on {}".format(self.algorithm.NAME, self.args.env_name),
        #                                      "{} + RND".format(self.algorithm.NAME),
        #                                      "red" if painter_label == 1 else "blue")
        # else:
        #     self.painter.plot_average_reward(return_list, 1,
        #                                      "{} on {}".format(self.algorithm.NAME, self.args.env_name),
        #                                      self.algorithm.NAME,
        #                                      "blue")
        #     plt.legend()
        #     self.painter.plot_episode_reward(return_list, 2,
        #                                      "{} on {}".format(self.algorithm.NAME, self.args.env_name),
        #                                      self.algorithm.NAME,
        #                                      "blue")
        # plt.legend()

        if painter_label == 1:
            self.painter.plot_average_reward(return_list, 1,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "{} + RND + NGU".format(self.algorithm.NAME),
                                             "red")
            plt.legend()
            self.painter.plot_episode_reward(return_list, 2,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "{} + RND + NGU".format(self.algorithm.NAME),
                                             "red")
        else:
            self.painter.plot_average_reward(return_list, 1,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "{} + RND".format(self.algorithm.NAME),
                                             "blue")
            plt.legend()
            self.painter.plot_episode_reward(return_list, 2,
                                             "{} on {}".format(self.algorithm.NAME, self.args.env_name),
                                             "{} + RND".format(self.algorithm.NAME),
                                             "blue")
        plt.legend()
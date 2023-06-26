# -*- coding = utf-8 -*-
# @time: 5/26/2023 10:15 PM
# Author: Yu Xia
# @File: Agent.py
# @software: PyCharm

import gymnasium
import torch
from matplotlib import pyplot as plt
from sympy import Piecewise
from tqdm import tqdm

from Algorithm import NGU
from Network import *
from Tools import Painter
from replaybuffer import *


def random_init(agent, cnn_model, env: gymnasium.Env, rnd: RNDNetwork):
    state, _ = env.reset()
    error = -100
    state = agent.preprocess(state).unsqueeze(0)
    state = cnn_model(state).squeeze(0)
    while abs(error) > 1:
        action = env.action_space.sample()
        s_prime, reward, done, _, _ = env.step(action)
        predict, target = rnd(state)
        error = rnd.get_intrinsic_reward(predict, target)

        state = s_prime
        state = agent.preprocess(state).unsqueeze(0)
        state = cnn_model(state).squeeze(0)


def segmentation_epsilon(all_rounds, train_rounds):
    """
    尚未完成，一种调节epsilon的分段函数。
    :param all_rounds:
    :param train_rounds:
    :return:
    """
    if train_rounds < all_rounds / 40:
        return 1
    elif all_rounds / 40 <= train_rounds < all_rounds / 2:
        return
    else:
        return 0.01


class Agent:
    def __init__(self, args, algorithm=None, env=None):
        self.algorithm = algorithm
        self.args = args
        self.painter = Painter()
        self.env = env
        self.rnd = RNDNetwork_CNN(args)
        self.rnd_weight = 0.4
        self.colors = ["red", "blue", "green", "yellow", "pink", "orange"]

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
        ])
        # self.exploration_coefficient = Piecewise(
        #     [
        #         (0, 1.0),
        #         (500, 0.1),
        #         (35000, 0.01)
        #     ], outside_value=0.01)

    def train_montezuma(self, RND=False, NGU=False, *, rnd_weight_decay=1.0, painter_label=1, PRE_PROCESS=True):
        env = gymnasium.make("ALE/MontezumaRevenge-v5")
        num_episodes = self.args.num_episodes

        cnn_model = CNN()
        RndNet = RNDNetwork(self.args)
        RND_WEIGHT = 1
        if RND:
            random_init(self, cnn_model, env, RndNet)

        return_list = []
        loss_list = []
        observations = []

        mean_acc = 0.0  # 运行时平均值
        var_acc = 0.0  # 运行时方差
        observation_count = 1
        only = False

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
                for episode in range(num_episodes // 10):
                    state, _ = env.reset()

                    if PRE_PROCESS:
                        """
                        将原始的输入通过转换为灰度图像并通过运行时标准差和均差进行归一化
                        最后将其裁剪到[-5, 5]范围内
                        """
                        state = self.preprocess(state).unsqueeze(0)
                        state = cnn_model(state).squeeze(0)
                        # observations.append(state)
                        #
                        # delta = state - mean_acc
                        # mean_acc += (delta / observation_count)
                        # var_acc += delta * (state - mean_acc)
                        #
                        # runtime_std_error = np.sqrt(var_acc.detach().numpy() / observation_count)
                        #
                        # state = (state - mean_acc.detach().numpy()) / runtime_std_error

                    done = False
                    episode_reward = 0
                    episode_loss = 0
                    while not done:
                        action = self.algorithm.select_action(state)
                        s_prime, reward, done, _, _ = env.step(action)

                        extrinsic_reward = reward / (max(abs(reward), 1))
                        if PRE_PROCESS:
                            s_prime = self.preprocess(s_prime).unsqueeze(0)
                            s_prime = cnn_model(s_prime).squeeze(0)
                            # observation_count += 1
                            # delta = s_prime - mean_acc
                            # mean_acc += (delta / observation_count)
                            # var_acc += delta * (s_prime - mean_acc)
                            #
                            # runtime_std_error = np.sqrt(var_acc.detach().numpy() / observation_count)
                            #
                            # s_prime = (s_prime - mean_acc.detach().numpy()) / runtime_std_error

                        if RND:
                            predict, target = RndNet(state)
                            intrinsic_reward = RndNet.get_intrinsic_reward(predict,
                                                                           target).item()  # / RndNet.running_std_error
                            # update_reward = 2 * extrinsic_reward + intrinsic_reward
                            update_reward = intrinsic_reward
                        else:
                            update_reward = extrinsic_reward

                        loss = self.algorithm.step(state.detach().numpy(), action, update_reward,
                                                   s_prime.detach().numpy(), done)

                        episode_reward += reward
                        episode_loss += loss

                        state = s_prime
                    if (episode_reward != 0 or num_episodes / 10 * i + episode + 1 > 50) and only:
                        env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode="human")
                        only = False
                    return_list.append(episode_reward)
                    loss_list.append(episode_loss)
                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{num_episodes / 10 * i + episode + 1}",
                                "return of last 10 rounds": f"{np.mean(return_list[-10:]):3f}",
                                "loss of last 10 rounds": f"{np.mean(loss_list[-10:]):3f}"
                            }
                        )
                    pbar.update(1)

    def collect_memories(self, RND=False):
        last_obs, _ = self.env.reset()
        last_obs = last_obs[105:180][:]
        for step in tqdm(range(self.algorithm.memory.learning_starts)):
            last_obs = self.preprocess(last_obs) / 255.
            cur_index = self.algorithm.memory.store_memory_obs(last_obs)
            encoded_obs = self.algorithm.memory.encoder_recent_observation()
            # choose action randomly
            action = self.env.action_space.sample()
            # interact with env
            obs, reward, done, info, _ = self.env.step(action)
            obs = obs[105:180][:]
            reward /= 100
            if RND:
                predict, target = self.rnd(encoded_obs)
                intrinsic_reward = self.rnd.get_intrinsic_reward(predict, target, CALC=False).item()
                update_reward = (1 - self.rnd_weight) * reward + self.rnd_weight * intrinsic_reward
            else:
                update_reward = reward

            # store other info
            self.algorithm.memory.store_memory_effect(cur_index, action, update_reward, done)

            if done:
                last_obs, _ = self.env.reset()
                last_obs = last_obs[105:180][:]

            last_obs = obs

    def train_breakout(self,
                       RND=False,
                       NGU=False,
                       *,
                       rnd_weight_decay=1.0,
                       painter_label=1,
                       PRE_PROCESS=True,
                       order=1,
                       test=False):

        env = gymnasium.make("ALE/Breakout-v5")
        self.env = env
        num_episodes = self.args.num_episodes
        if test:
            self.algorithm.epsilon = 0.001

        save_model_freq = num_episodes // 5

        loss_list = []

        self.algorithm.memory_reset()
        self.collect_memories(RND)
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:

                return_list = []
                for episode in range(num_episodes // 10):
                    state, info = env.reset()

                    episode_reward = 0
                    episode_loss = 0

                    done = False
                    while not done:
                        state = self.preprocess(state)
                        current_index = self.algorithm.memory.store_memory_obs(state)
                        encoded_obs = self.algorithm.memory.encoder_recent_observation()
                        # # encoded_obs = torch.from_numpy(encoded_obs).unsqueeze(0).to(torch.float32) / 255.0
                        # encoded_obs = self.preprocess(encoded_obs)
                        action = self.algorithm.select_action(encoded_obs)
                        s_prime, reward, done, _, _ = env.step(action)

                        if RND:
                            predict, target = self.rnd(encoded_obs)
                            intrinsic_reward = self.rnd.get_intrinsic_reward(predict, target).item()

                            update_reward = (1 - self.rnd_weight) * reward + self.rnd_weight * intrinsic_reward
                        else:
                            update_reward = reward

                        if self.rnd_weight > 0.001:
                            self.rnd_weight *= 0.9995

                        loss = self.algorithm.step(current_index, action, update_reward, done)

                        episode_reward += reward
                        episode_loss += loss

                        state = s_prime

                    return_list.append(episode_reward)
                    loss_list.append(episode_loss)
                    if self.algorithm.epsilon > 0.01 and not test:
                        # self.algorithm.epsilon *= 0.997
                        # if np.mean(return_list[-10:]) < 10.0 and self.algorithm.epsilon < 0.2:
                        #     self.algorithm.epsilon = 0.2
                        self.algorithm.epsilon = 1 / (1 + np.log2(np.mean(return_list[-10:]) + 1))

                    # self.painter.plot_average_reward(episode_reward,
                    #                                  window=1,
                    #                                  title="{} on breakout".format(self.algorithm.NAME),
                    #                                  curve_label="{}".format(
                    #                                      self.algorithm.NAME + "RND" if RND else ""),
                    #                                  color=self.colors[order - 1])

                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{num_episodes / 10 * i + episode + 1}",
                                "return of last 10 rounds": f"{np.mean(return_list[-10:]):3f}",
                                "loss of last 10 rounds": f"{np.mean(loss_list[-10:]):9f}",
                                "exploration": f"{self.algorithm.epsilon:6f}",
                                "frame count": f"{info}"
                            }
                        )
                        self.painter.plot_average_reward_by_list(return_list,
                                                                 window=1,
                                                                 title="{} on breakout".format(self.algorithm.NAME),
                                                                 curve_label="{}".format(
                                                                     self.algorithm.NAME + "RND" if RND else ""),
                                                                 color=self.colors[order - 1]
                                                                 )
                        return_list = []

                    if (num_episodes / 10 * i + episode + 1) % save_model_freq == 0 and not test:
                        torch.save({"main_net_state_dict": self.algorithm.main_net.state_dict(),
                                    "target_net_state_dict": self.algorithm.target_net.state_dict()},
                                   "./checkpoint/{}_model_breakout_{}_{}.pth".format(self.algorithm.NAME,
                                                                                     num_episodes / 10 * i + episode + 1,
                                                                                     "T" if RND else "F"))

                    pbar.update(1)

        plt.legend()
        self.painter.plot_episode_reward(return_list,
                                         window=2,
                                         title="{} on breakout".format(self.algorithm.NAME),
                                         curve_label="{}".format(self.algorithm.NAME + "RND" if RND else ""),
                                         color=self.colors[order - 1])

    def train_RoadRunner(self, RND=False,
                         NGU=False,
                         *,
                         rnd_weight_decay=1.0,
                         painter_label=1,
                         PRE_PROCESS=True,
                         order=1,
                         test=False):

        env = gymnasium.make("ALE/RoadRunner-v5")
        self.env = env
        num_episodes = self.args.num_episodes
        if test:
            self.algorithm.epsilon = 0.001

        save_model_freq = num_episodes // 20

        all_return_list = []
        return_list = []
        loss_list = []
        if test:
            env = gymnasium.make("ALE/RoadRunner-v5", render_mode="human")
        else:
            self.algorithm.memory_reset()
            self.collect_memories(RND)

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
                for episode in range(num_episodes // 10):
                    state, info = env.reset()
                    state = state[105: 180][:]

                    episode_reward = 0
                    episode_loss = 0

                    done = False
                    while not done:
                        state = self.preprocess(state) / 255.
                        # grayImage = np.transpose(state.detach().cpu().numpy(), (1, 2, 0))
                        # plt.imshow(grayImage, cmap="gray")
                        # plt.axis('off')
                        # plt.show()
                        current_index = self.algorithm.memory.store_memory_obs(state)
                        encoded_obs = self.algorithm.memory.encoder_recent_observation()
                        # encoded_obs = self.preprocess(encoded_obs).to(self.algorithm.device) / 255.
                        action = self.algorithm.select_action(encoded_obs)
                        s_prime, reward, done, _, _ = env.step(action)
                        s_prime = s_prime[105:180][:]
                        reward /= 100
                        # if reward == 0:
                        #     reward = 0.01

                        if RND:
                            predict, target = self.rnd(encoded_obs)
                            intrinsic_reward = self.rnd.get_intrinsic_reward(predict, target).item()

                            update_reward = (1 - self.rnd_weight) * reward + self.rnd_weight * intrinsic_reward
                        else:
                            update_reward = reward

                        # if self.rnd_weight > 0.001:
                        #     self.rnd_weight *= 0.9995

                        loss = self.algorithm.step(current_index, action, update_reward, done)

                        episode_reward += reward
                        episode_loss += loss

                        state = s_prime

                    all_return_list.append(episode_reward)
                    return_list.append(episode_reward)
                    loss_list.append(episode_loss)

                    if self.algorithm.epsilon > 0.01 and len(self.painter.return_list) > 0:
                        # self.algorithm.epsilon = 1 / (1 + ((np.log((np.mean(all_return_list[-10:])))) + 1))
                        if num_episodes / 10 * i + episode + 1 < num_episodes / 10:
                            self.algorithm.epsilon *= 0.9995
                        elif num_episodes / 10 * i + episode + 1 < num_episodes / 2:
                            self.algorithm.epsilon = 0.1
                        else:
                            self.algorithm.epsilon = 0.01

                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{num_episodes / 10 * i + episode + 1}",
                                "return of last 10 rounds": f"{np.mean(all_return_list[-10:]):3f}",
                                "loss of last 10 rounds": f"{np.mean(loss_list[-10:]):9f}",
                                "exploration": f"{self.algorithm.epsilon:6f}",
                                "info": f"{info}"
                            }
                        )
                        self.painter.plot_average_reward_by_list(return_list,
                                                                 window=1,
                                                                 title="{} on RoadRunner".format(self.algorithm.NAME),
                                                                 curve_label="{}".format(
                                                                     self.algorithm.NAME + "RND" if RND else ""),
                                                                 color=self.colors[order - 1]
                                                                 )
                        return_list = []
                    if (num_episodes / 10 * i + episode + 1) % save_model_freq == 0:
                        torch.save({"main_net_state_dict": self.algorithm.main_net.state_dict(),
                                    "target_net_state_dict": self.algorithm.target_net.state_dict()},
                                   "./checkpoint/{}_model_RoadRunner_{}_{}.pth".format(self.algorithm.NAME,
                                                                                       (
                                                                                               num_episodes / 10 * i + episode + 1),
                                                                                       "T" if RND else "F"))
                    pbar.update(1)

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
                                alpha_t = 1 + ((
                                                       RndNet.sum_error - RndNet.running_mean_deviation) / RndNet.running_std_error)
                                ngu_reward = ngu_reward * min(max(alpha_t, 1), ngu.L)

                                normalized_reward = ((ngu_reward - min_reward) / (
                                        max_reward - min_reward)) * 0.25  # 归一化
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


"""
----- ----------------------------------以下是做实验的代码----------------------------------
"""


class Agent_Experiment:
    def __init__(self, args, algorithm=None, env=None):
        self.algorithm = algorithm
        self.args = args
        self.painter = Painter()
        self.env = env
        self.rnd = RNDNetwork_CNN(args)
        self.rnd_weight = 0.4
        self.colors = ["red", "blue", "green", "yellow", "pink", "orange"]

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
        ])

        self.state_count = 0
        self.state_judge = None
        self.action_judge = None
        self.get_intrinsic_reward_count = 0  # 计算得到过多少次内在奖励
        self.state_list = []
        self.super_list = []

    def get_exploration_cofficient(self, x, N):
        if x < N / 40:
            return 1
        elif N / 40 <= x < N / 2:
            epsilon = 1 - 0.9 * (x - N / 40) / (N / 2 - N / 40)
            return epsilon
        else:
            return 0.01

    def get_intrinsic_cofficient(self, x, N):
        if x < N / 40:
            return 1
        elif N / 40 <= x < N / 2:
            epsilon = 1 - 0.9 * (x - N / 40) / (N / 2 - N / 40)
            return epsilon
        else:
            return 0.01

    def Specify_State(self, state, action, state_count=11):
        if self.state_count < state_count:
            self.state_count += 1
        elif self.state_count == state_count:
            self.state_judge = torch.tensor(state, dtype=torch.float).to(self.algorithm.device)
            self.action_judge = action
            self.state_count += 1
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.algorithm.device)
            if self.algorithm.loss_func(self.state_judge, state) < 0.1 and self.get_intrinsic_reward_count % 10 == 0:
                q_value = self.algorithm.main_net(state.unsqueeze(0)).squeeze(0)[action]
                super_value = self.algorithm.super_net(state.unsqueeze(0)).squeeze(0)[action]
                self.state_list.append(q_value.detach().cpu().numpy())
                self.super_list.append(super_value.detach().cpu().numpy())
                self.painter.plot_NetOutput(self.state_list, label="state", color="red")
                self.painter.plot_NetOutput(self.super_list, label="super", color="blue")
                if len(self.state_list) == 1:
                    plt.legend()

    def collect_memories(self, RND=False):
        last_obs, _ = self.env.reset()
        last_obs = last_obs[105:180][:]
        for step in tqdm(range(self.algorithm.memory.learning_starts)):
            last_obs = self.preprocess(last_obs) / 255.
            cur_index = self.algorithm.memory.store_memory_obs(last_obs)
            encoded_obs = self.algorithm.memory.encoder_recent_observation()
            # choose action randomly
            action = self.env.action_space.sample()
            # interact with env
            obs, reward, done, info, _ = self.env.step(action)
            obs = obs[105:180][:]
            reward /= 100
            if RND:
                predict, target = self.rnd(encoded_obs)
                intrinsic_reward = self.rnd.get_intrinsic_reward(predict, target, CALC=False).item()
                update_reward = (1 - self.rnd_weight) * reward + self.rnd_weight * intrinsic_reward
            else:
                update_reward = reward

            # store other info
            self.algorithm.memory.store_memory_effect(cur_index, action, update_reward, done)

            if done:
                last_obs, _ = self.env.reset()
                last_obs = last_obs[105:180][:]

            last_obs = obs

    def train_RoadRunner(self, RND=False,
                         NGU=False,
                         *,
                         rnd_weight_decay=1.0,
                         painter_label=1,
                         use_super=False,
                         order=1,
                         test=False):

        env = gymnasium.make("ALE/RoadRunner-v5")
        self.env = env
        num_episodes = self.args.num_episodes
        if test:
            self.algorithm.epsilon = 0.001

        save_model_freq = num_episodes // 20

        all_return_list = []
        return_list = []
        loss_list = []
        if test:
            env = gymnasium.make("ALE/RoadRunner-v5", render_mode="human")
        else:
            self.algorithm.memory_reset()
            # self.collect_memories(RND)

        param1 = list(self.algorithm.main_net.parameters())
        param2 = list(self.algorithm.super_net.model.parameters())
        print("主网络和super网络的参数是否一致:{}".format(all(torch.equal(p1, p2) for p1, p2 in zip(param1, param2))))

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
                for episode in range(num_episodes // 10):
                    state, info = env.reset()
                    state = state[105: 180][:]

                    episode_reward = 0
                    episode_loss = 0

                    done = False
                    while not done:
                        state = self.preprocess(state) / 255.
                        # grayImage = np.transpose(state.detach().cpu().numpy(), (1, 2, 0))
                        # plt.imshow(grayImage, cmap="gray")
                        # plt.axis('off')
                        # plt.show()
                        current_index = self.algorithm.memory.store_memory_obs(state)
                        encoded_obs = self.algorithm.memory.encoder_recent_observation()
                        # encoded_obs = self.preprocess(encoded_obs).to(self.algorithm.device) / 255.
                        action = self.algorithm.select_action(encoded_obs)
                        s_prime, reward, done, _, _ = env.step(action)
                        s_prime = s_prime[105:180][:]
                        reward /= 100
                        # if reward == 0:
                        #     reward = 0.01

                        if RND:
                            predict, target = self.rnd(encoded_obs)
                            intrinsic_reward = self.rnd.get_intrinsic_reward(predict, target).item()

                            update_reward = (1 - self.rnd_weight) * reward + self.rnd_weight * intrinsic_reward
                        elif use_super:
                            self.get_intrinsic_reward_count += 1
                            intrinsic_reward = self.algorithm.get_super_reward(encoded_obs, action, reward)
                            self.Specify_State(encoded_obs, action)
                            update_reward = 0.6 * reward + 0.4 * intrinsic_reward
                        else:
                            update_reward = reward

                        # if self.rnd_weight > 0.001:
                        #     self.rnd_weight *= 0.9995

                        loss = self.algorithm.step(current_index, encoded_obs, action, update_reward, done)

                        episode_reward += reward
                        episode_loss += loss

                        state = s_prime

                    all_return_list.append(episode_reward)
                    return_list.append(episode_reward)
                    loss_list.append(episode_loss)

                    if self.algorithm.epsilon >= 0.01 and len(self.painter.return_list) > 0:
                        # self.algorithm.epsilon = 1 / (1 + ((np.log((np.mean(all_return_list[-10:])))) + 1))
                        # if num_episodes / 10 * i + episode + 1 < num_episodes / 10:
                        #     self.algorithm.epsilon *= 0.9995
                        # elif num_episodes / 10 * i + episode + 1 < num_episodes / 2:
                        #     self.algorithm.epsilon = 0.1
                        # else:
                        #     self.algorithm.epsilon = 0.01
                        self.algorithm.epsilon = self.get_exploration_cofficient(num_episodes / 10 * i + episode + 1,
                                                                                 num_episodes)

                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{num_episodes / 10 * i + episode + 1}",
                                "return of last 10 rounds": f"{np.mean(all_return_list[-10:]):3f}",
                                "loss of last 10 rounds": f"{np.mean(loss_list[-10:]):9f}",
                                "exploration": f"{self.algorithm.epsilon:6f}",
                                "info": f"{info}"
                            }
                        )
                        self.painter.plot_average_reward_by_list(return_list,
                                                                 window=1,
                                                                 title="{} on RoadRunner".format(self.algorithm.NAME),
                                                                 curve_label="{}".format(
                                                                     self.algorithm.NAME + "Super" if use_super else ""),
                                                                 color=self.colors[order - 1]
                                                                 )
                        return_list = []
                    if (num_episodes / 10 * i + episode + 1) % save_model_freq == 0:
                        torch.save({"main_net_state_dict": self.algorithm.main_net.state_dict(),
                                    "target_net_state_dict": self.algorithm.target_net.state_dict()},
                                   "./checkpoint/{}_model_RoadRunner_{}_{}.pth".format(self.algorithm.NAME,
                                                                                       (
                                                                                               num_episodes / 10 * i + episode + 1),
                                                                                       "T" if RND else "F"))
                    pbar.update(1)
        self.painter.plot_average_reward_by_list(return_list,
                                                 window=1,
                                                 end=True,
                                                 title="{} on RoadRunner".format(self.algorithm.NAME),
                                                 curve_label="{}".format(
                                                     self.algorithm.NAME + "Super" if use_super else ""),
                                                 color=self.colors[order - 1],
                                                 saveName="Train result for {} rounds on RoadRunner".format(
                                                     num_episodes))

    def train_breakout(self,
                       RND=False,
                       use_super=False,
                       *,
                       rnd_weight_decay=1.0,
                       painter_label=1,
                       PRE_PROCESS=True,
                       order=1,
                       test=False):

        env = gymnasium.make("ALE/Breakout-v5")
        self.env = env
        num_episodes = self.args.num_episodes
        if test:
            self.algorithm.epsilon = 0.001

        save_model_freq = num_episodes // 5

        loss_list = []
        return_list = []

        if test:
            pass
        else:
            self.algorithm.memory_reset()
            # self.collect_memories(RND)
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:

                for episode in range(num_episodes // 10):
                    state, info = env.reset()
                    state = state[20:, :]

                    episode_reward = 0
                    episode_loss = 0

                    done = False
                    while not done:
                        state = self.preprocess(state)
                        current_index = self.algorithm.memory.store_memory_obs(state)
                        encoded_obs = self.algorithm.memory.encoder_recent_observation()
                        # # encoded_obs = torch.from_numpy(encoded_obs).unsqueeze(0).to(torch.float32) / 255.0
                        # encoded_obs = self.preprocess(encoded_obs)
                        action = self.algorithm.select_action(encoded_obs)
                        s_prime, reward, done, _, _ = env.step(action)
                        s_prime = s_prime[20:, :]

                        if RND:
                            predict, target = self.rnd(encoded_obs)
                            intrinsic_reward = self.rnd.get_intrinsic_reward(predict, target).item()

                            update_reward = (1 - self.rnd_weight) * reward + self.rnd_weight * intrinsic_reward
                        elif use_super:
                            self.get_intrinsic_reward_count += 1
                            intrinsic_reward = self.algorithm.get_super_reward(encoded_obs, action, reward)
                            # self.Specify_State(encoded_obs, action)
                            update_reward = 0.6 * reward + 0.4 * intrinsic_reward
                        else:
                            update_reward = reward

                        if self.rnd_weight > 0.001:
                            self.rnd_weight *= 0.9995

                        loss = self.algorithm.step(current_index, state, action, update_reward, done)

                        episode_reward += reward
                        episode_loss += loss

                        state = s_prime

                    return_list.append(episode_reward)
                    loss_list.append(episode_loss)
                    if self.algorithm.epsilon > 0.01 and not test:
                        # self.algorithm.epsilon *= 0.997
                        # if np.mean(return_list[-10:]) < 10.0 and self.algorithm.epsilon < 0.2:
                        #     self.algorithm.epsilon = 0.2
                        self.algorithm.epsilon = self.get_exploration_cofficient(num_episodes / 10 * i + episode + 1,
                                                                                 num_episodes)

                    # self.painter.plot_average_reward(episode_reward,
                    #                                  window=1,
                    #                                  title="{} on breakout".format(self.algorithm.NAME),
                    #                                  curve_label="{}".format(
                    #                                      self.algorithm.NAME + "RND" if RND else ""),
                    #                                  color=self.colors[order - 1])

                    if (episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": f"{num_episodes / 10 * i + episode + 1}",
                                "return of last 10 rounds": f"{np.mean(return_list[-10:]):3f}",
                                "loss of last 10 rounds": f"{np.mean(loss_list[-10:]):9f}",
                                "exploration": f"{self.algorithm.epsilon:6f}",
                                "frame count": f"{info}"
                            }
                        )
                    if len(return_list) == 20:
                        self.painter.plot_average_reward_by_list(return_list,
                                                                 window=1,
                                                                 title="{} on breakout".format(self.algorithm.NAME),
                                                                 curve_label="{}".format(
                                                                     self.algorithm.NAME + "RND" if RND else ""),
                                                                 color=self.colors[order - 1]
                                                                 )
                        return_list = []

                    if (num_episodes / 10 * i + episode + 1) % save_model_freq == 0 and not test:
                        torch.save({"main_net_state_dict": self.algorithm.main_net.state_dict(),
                                    "target_net_state_dict": self.algorithm.target_net.state_dict()},
                                   "./checkpoint/{}_model_breakout_{}_{}.pth".format(self.algorithm.NAME,
                                                                                     num_episodes / 10 * i + episode + 1,
                                                                                     "T" if RND else "F"))

                    pbar.update(1)

        self.painter.plot_average_reward_by_list(return_list,
                                                 window=1,
                                                 end=True,
                                                 title="{} on RoadRunner".format(self.algorithm.NAME),
                                                 curve_label="{}".format(
                                                     self.algorithm.NAME + "Super" if use_super else ""),
                                                 color=self.colors[order - 1],
                                                 saveName="Train result for {} rounds on breakout".format(
                                                     num_episodes))

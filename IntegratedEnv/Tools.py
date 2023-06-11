# -*- coding = utf-8 -*-
# @time: 5/26/2023 10:38 PM
# Author: Yu Xia
# @File: Tools.py
# @software: PyCharm
import argparse
import matplotlib.pyplot as plt
import numpy as np


class SetupArgs:
    def __init__(self):
        pass

    def get_args(self, description="Parameters setting"):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--num_episodes', type=int, default=1500, help='Training frequency')
        parser.add_argument('--seed', type=int, default=996, metavar='S', help='set random seed')
        parser.add_argument("--gamma", type=float, default=0.99, metavar='S', help='discounted rate')
        parser.add_argument('--epsilon', type=float, default=1, metavar='S', help='Exploration rate')
        parser.add_argument('--env_name', type=str, default="MountainCar-v0", metavar='S', help="Environment name")

        return parser.parse_args()


class Painter:
    def __init__(self):
        pass

    def plot_average_reward(self, reward_list, window, title, curve_label, color, end=False, xlabel="Episodes", ylabel="Average reward"):
        """
        计算并绘制前n个回合的平均奖励并更新
        """
        plt.figure(window)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        list_t = [np.mean(reward_list[:i]) for i in range(len(reward_list))]

        plt.plot(np.array(list_t), color=color, label=curve_label, linewidth=0.8)

        if end:
            plt.show()
        else:
            plt.pause(0.001)

    def plot_episode_reward(self, reward_list, window, title, curve_label, color, end=False, xlabel="Episodes", ylabel="Returns"):
        plt.figure(window)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        list_t = np.array(reward_list)
        plt.plot(np.array(list_t), color=color, label=curve_label, linewidth=0.7)

        if end:
            plt.show()
        else:
            plt.pause(0.001)

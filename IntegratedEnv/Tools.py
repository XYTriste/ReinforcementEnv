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
        parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
        parser.add_argument('--num_episodes', type=int, default=1000, help='Training frequency')
        parser.add_argument('--seed', type=int, default=21, metavar='S', help='set random seed')
        parser.add_argument("--gamma", type=float, default=0.99, metavar='S', help='discounted rate')
        parser.add_argument('--epsilon', type=float, default=0.1, metavar='S', help='Exploration rate')
        parser.add_argument('--env_name', type=str, default="MountainCar-v0", metavar='S', help="Environment name")

        return parser.parse_args()

class Painter:
    def __init__(self):
        pass

    def plot_reward(self, reward_list, window, title, end=False, xlabel="Episodes", ylabel="Returns"):
        plt.figure(window)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        list_t = [np.mean(reward_list[:i]) for i in range(len(reward_list))]
        plt.plot(np.array(list_t))

        if end:
            plt.show()
        else:
            plt.pause(0.001)
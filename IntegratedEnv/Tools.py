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
        parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
        parser.add_argument('--num_episodes', type=int, default=1500, help='Training frequency')
        parser.add_argument('--seed', type=int, default=24, metavar='S', help='set random seed')
        parser.add_argument("--gamma", type=float, default=0.95, metavar='S', help='discounted rate')
        parser.add_argument('--epsilon', type=float, default=1, metavar='S', help='Exploration rate')
        parser.add_argument('--buffer_size', type=int, default=2 ** 16, metavar='S', help='Experience replay buffer size')
        parser.add_argument('--env_name', type=str, default="MountainCar-v0", metavar='S', help="Environment name")

        return parser.parse_args()


class Painter:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.return_list = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.x_data, self.y_data)

        self.average_data = []
        self.data_count = 0

    def plot_average_reward(self, reward, window, title, curve_label, color, end=False, xlabel="Episodes", ylabel="Average reward"):
        """
        计算并绘制前n个回合的平均奖励并更新
        """
        plt.figure(window)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # list_t = [np.mean(reward_list[:i]) for i in range(len(reward_list))]
        if len(self.return_list) == 0:
            self.return_list.append(reward)
        else:
            size = len(self.return_list) + 1
            new_data = self.return_list[-1] + (reward - self.return_list[-1]) / size
            self.return_list.append(new_data)
        plt.plot(np.array(self.return_list), color=color, label=curve_label, linewidth=0.8)

        if end:
            plt.show()
        else:
            plt.pause(0.001)

    def plot_average_reward_by_list(self, list, window, title, curve_label, color, end=False, xlabel="steps",
                            ylabel="Average return", saveName="default_name"):
        plt.ion()
        plt.figure(window)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if end:
            plt.ioff()
            plt.savefig('./train_pic/' + saveName + '.png')
            return
        # for reward in list:
        #     if len(self.return_list) == 0:
        #         self.return_list.append(reward)
        #     else:
        #         size = len(self.return_list) + 1
        #         new_data = self.return_list[-1] + (reward - self.return_list[-1]) / size
        #         self.return_list.append(new_data)
        average_data = np.average(list)
        if len(self.return_list) == 0:
            self.return_list.append(average_data)
        else:
            new_data = self.return_list[-1] + (average_data - self.return_list[-1]) / len(self.return_list)
            self.return_list.append(new_data)
        plt.plot(np.array(self.return_list), color=color, label=curve_label, linewidth=0.8)
        plt.pause(0.05)

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

    def plot_reward_test(self, reward):
        self.x_data.append(len(self.x_data))
        self.y_data.append(reward)

        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        plt.pause(0.001)

    def plot_NetOutput(self, list, label=None, color="red"):
        plt.figure("state")
        plt.xlabel('steps')
        plt.ylabel('q_value')
        plt.plot(list, label=label, color=color)
        plt.pause(0.01)



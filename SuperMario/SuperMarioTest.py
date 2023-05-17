# -*- coding = utf-8 -*-
# @time: 5/15/2023 4:45 PM
# Author: Yu Xia
# @File: SuperMarioTest.py
# @software: PyCharm

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

INPUT_DIM = (240, 256, 3)

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.5
GAMMA = 0.95
TARGET_REPLACE_ITER = 100  # 目标网络的更新速率，100指的是每更新当前网络100次则更新一次目标网络
MEMORY_CAPACITY = 2000

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        c, width, height = INPUT_DIM

        self.commonLayer = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
            # nn.Linear(3136, 512),
            # nn.ReLU(),
            # nn.Linear(512, output_dim)
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = self.out(x)

        return actions_value


if __name__ == '__main__':
    done = True
    env.reset()
    for step in range(5000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            state = env.reset()

    env.close()

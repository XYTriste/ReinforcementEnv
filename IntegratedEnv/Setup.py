# -*- coding = utf-8 -*-
# @time: 5/26/2023 8:52 PM
# Author: Yu Xia
# @File: Setup.py.py
# @software: PyCharm
from Tools import *
from Algorithm import *
import gymnasium as gym
from Agent import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = SetupArgs().get_args()

    env = gym.make(args.env_name, render_mode="rgb_array")
    dqn = DQN(args)

    agent = Agent(args, env, dqn)
    agent.train()

    plt.show()

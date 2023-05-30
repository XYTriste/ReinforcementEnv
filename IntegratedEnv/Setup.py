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
    dqn_first = DQN(args)
    dqn_second = DQN(args)

    dqn_second.main_net.load_state_dict(dqn_first.main_net.state_dict())
    dqn_second.target_net.load_state_dict(dqn_first.target_net.state_dict())

    agent_useRnd = Agent(args, env, dqn_first)
    agent_unuseRnd = Agent(args, env, dqn_second)

    agent_useRnd.train()
    agent_unuseRnd.train(False)

    plt.show()

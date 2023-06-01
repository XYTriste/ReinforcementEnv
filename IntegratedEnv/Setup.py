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


def mountaincar_DQN():
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


def blackjack_actor_critic():
    args = SetupArgs().get_args()
    args.env_name = "Blackjack-v1"
    args.num_episodes = 20000

    env = gym.make(args.env_name, render_mode="rgb_array")

    a2c = Actor_Critic(args)

    agent = Agent(args, env, a2c)
    agent.train(use_rnd=False)

    plt.show()


if __name__ == "__main__":
    blackjack_actor_critic()

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

    args.INPUT_DIM = 2
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 3
    args.HIDDEN_DIM_NUM = 3
    args.SIZEOF_EVERY_MEMORY = 7
    args.MEMORY_SHAPE = (2, 1, 1, 2, 1)

    env = gym.make(args.env_name, render_mode="rgb_array")
    dqn_first = DQN(args)
    dqn_second = DQN(args)

    dqn_second.main_net.load_state_dict(dqn_first.main_net.state_dict())
    dqn_second.target_net.load_state_dict(dqn_first.target_net.state_dict())

    agent_useRnd = Agent(args, env, dqn_first)
    agent_unuseRnd = Agent(args, env, dqn_second)

    agent_useRnd.train(use_rnd=True, use_ngu=True,rnd_weight_decay=0.95)
    # agent_unuseRnd.train(rnd_weight_decay=0.95)

    plt.show()


def blackjack_actor_critic():
    args = SetupArgs().get_args()
    args.env_name = "Blackjack-v1"
    args.num_episodes = 20000

    env = gym.make(args.env_name, render_mode="rgb_array")

    ac = Actor_Critic(args)

    agent = Agent(args, env, ac)
    agent.train(use_rnd=False)

    plt.show()


if __name__ == "__main__":
    mountaincar_DQN()

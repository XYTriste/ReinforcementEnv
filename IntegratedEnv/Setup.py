# -*- coding = utf-8 -*-
# @time: 5/26/2023 8:52 PM
# Author: Yu Xia
# @File: Setup.py.py
# @software: PyCharm
import torch

from Algorithm import *
from Agent import *
import matplotlib.pyplot as plt


def mountaincar_DQN():
    args = SetupArgs().get_args()

    args.num_episodes = 1500
    args.INPUT_DIM = 2
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 3
    args.HIDDEN_DIM_NUM = 3
    args.SIZEOF_EVERY_MEMORY = 7
    args.MEMORY_SHAPE = (2, 1, 1, 2, 1)

    env = gym.make(args.env_name, render_mode="rgb_array")
    dqn_first = DQN(args, NAME="DDQN")
    dqn_second = DQN(args, NAME="DQN")

    dqn_second.main_net.load_state_dict(dqn_first.main_net.state_dict())
    dqn_second.target_net.load_state_dict(dqn_first.target_net.state_dict())

    agent_useRnd = Agent(args, dqn_first)
    agent_unuseRnd = Agent(args, dqn_second)

    agent_useRnd.train(use_rnd=True, use_ngu=False, rnd_weight_decay=0.95, painter_label=1)
    agent_unuseRnd.train(use_rnd=True, use_ngu=False, rnd_weight_decay=0.95, painter_label=2)

    plt.show()


def blackjack_actor_critic():
    args = SetupArgs().get_args()
    args.env_name = "Blackjack-v1"
    args.num_episodes = 20000
    args.INPUT_DIM = 2
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 2
    args.HIDDEN_DIM_NUM = 3
    args.SIZEOF_EVERY_MEMORY = 7
    args.MEMORY_SHAPE = (2, 1, 1, 2, 1)

    env = gym.make(args.env_name, render_mode="rgb_array")

    ac = Actor_Critic(args)

    agent = Agent(args, ac)
    agent.train(use_rnd=False)

    plt.show()


def montezuma_revenge():
    args = SetupArgs().get_args()

    args.num_episodes = 2000
    args.INPUT_DIM = 512
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 18
    args.HIDDEN_DIM_NUM = 5
    args.SIZEOF_EVERY_MEMORY = 1027
    args.MEMORY_SHAPE = (512, 1, 1, 512, 1)

    double_dqn = DQN(args, NAME="DDQN")
    agent = Agent(args, double_dqn)
    agent.train_montezuma(RND=True)


def breakout():
    args = SetupArgs().get_args()

    args.num_episodes = 30000
    args.INPUT_DIM = 512
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 4
    args.HIDDEN_DIM_NUM = 5
    args.SIZEOF_EVERY_MEMORY = 1027
    args.MEMORY_SHAPE = (512, 1, 1, 512, 1)

    double_dqn = DQN(args, NAME="DDQN")
    agent = Agent(args, double_dqn)
    agent.train_breakout()

    torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
                "target_net_state_dict": double_dqn.target_net.state_dict()}, "ddqn_model_breakout.pth")


if __name__ == "__main__":
    breakout()

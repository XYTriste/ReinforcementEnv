# -*- coding = utf-8 -*-
# @time: 5/26/2023 8:52 PM
# Author: Yu Xia
# @File: Setup.py.py
# @software: PyCharm
import copy

import torch
from Algorithm import *
from Agent import *
import matplotlib.pyplot as plt
from labml import experiment
from labml.internal.configs.dynamic_hyperparam import FloatDynamicHyperParam, IntDynamicHyperParam
from datetime import datetime
import os
from PPO_algorithm import *


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

    env = gymnasium.make(args.env_name, render_mode="rgb_array")

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

    args.num_episodes = 2000
    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 4
    args.HIDDEN_DIM_NUM = 5

    # checkpoint = torch.load('./checkpoint/DQN_model_breakout_450.0.pth')

    dqn = DQN_CNN(args)
    double_dqn = DQN_CNN(args, NAME="DDQN")
    double_dqn.main_net.load_state_dict(dqn.main_net.state_dict())
    double_dqn.target_net.load_state_dict(dqn.target_net.state_dict())
    # double_dqn.main_net.load_state_dict(checkpoint['main_net_state_dict'])
    # double_dqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent1 = Agent(args, dqn)
    agent1.train_breakout()

    agent2 = Agent(args, double_dqn)
    agent2.train_breakout(RND=True, order=2)

    torch.save({"main_net_state_dict": dqn.main_net.state_dict(),
                "target_net_state_dict": dqn.target_net.state_dict()}, "checkpoint/dqn_model_breakout_final.pth")
    torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
                "target_net_state_dict": double_dqn.target_net.state_dict()},
               "checkpoint/DDQN_model_breakout_final.pth")
    plt.show()


def RoadRunner():
    args = SetupArgs().get_args()

    args.num_episodes = 20000
    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 18
    args.HIDDEN_DIM_NUM = 5

    dqn_checkpoint = torch.load('./checkpoint/DQN_model_RoadRunner_8000.0_F.pth')

    dqn = DQN_CNN(args)
    double_dqn = DQN_CNN(args, NAME="DDQN")

    # double_dqn.main_net.load_state_dict(dqn.main_net.state_dict())
    # double_dqn.target_net.load_state_dict(dqn.target_net.state_dict())
    # double_dqn.main_net.load_state_dict(dqn_checkpoint['main_net_state_dict'])
    # double_dqn.target_net.load_state_dict(dqn_checkpoint['target_net_state_dict'])
    dqn.main_net.load_state_dict(dqn_checkpoint['main_net_state_dict'])
    dqn.target_net.load_state_dict(dqn_checkpoint['target_net_state_dict'])
    agent1 = Agent(args, dqn)
    agent1.train_RoadRunner(test=True)
    torch.save({"main_net_state_dict": dqn.main_net.state_dict(),
                "target_net_state_dict": dqn.target_net.state_dict()},
               "checkpoint/dqn_model_RoadRunner_70000_final.pth")

    # agent2 = Agent(args, double_dqn)
    # agent2.train_RoadRunner(RND=False, order=2)
    #
    # torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
    #             "target_net_state_dict": double_dqn.target_net.state_dict()},
    #            "checkpoint/DDQN_model_RoadRunner_final.pth")
    plt.show()


def RoadRunner_Experiment():
    args = SetupArgs().get_args()

    args.num_episodes = 10000
    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 18
    args.HIDDEN_DIM_NUM = 5

    # dqn_checkpoint = torch.load('./checkpoint/DQN_model_RoadRunner_8000.0_F.pth')

    dqn = DQN_CNN_Super(args)
    dqnCopy = copy.deepcopy(dqn)
    # double_dqn = DQN_CNN_Super(args, NAME="DDQN")

    # double_dqn.main_net.load_state_dict(dqn.main_net.state_dict())
    # double_dqn.target_net.load_state_dict(dqn.target_net.state_dict())
    # double_dqn.main_net.load_state_dict(dqn_checkpoint['main_net_state_dict'])
    # double_dqn.target_net.load_state_dict(dqn_checkpoint['target_net_state_dict'])
    # dqn.main_net.load_state_dict(dqn_checkpoint['main_net_state_dict'])
    # dqn.target_net.load_state_dict(dqn_checkpoint['target_net_state_dict'])
    agent1 = Agent_Experiment(args, dqn)
    agent1.train_RoadRunner(use_super=True)
    agent2 = Agent_Experiment(args, dqnCopy)
    agent2.train_RoadRunner(use_super=False, order=2)
    torch.save({"main_net_state_dict": dqn.main_net.state_dict(),
                "target_net_state_dict": dqn.target_net.state_dict()},
               "checkpoint/dqn_super_model_RoadRunner_{}_final.pth".format(args.num_episodes))

    # agent2 = Agent(args, double_dqn)
    # agent2.train_RoadRunner(RND=False, order=2)
    #
    torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
                "target_net_state_dict": double_dqn.target_net.state_dict()},
               "checkpoint/dqn_model_RoadRunner_{}_final.pth".format(args.num_episodes))
    plt.show()


def breakout_experiment():
    args = SetupArgs().get_args()

    args.num_episodes = 15000
    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 4
    args.HIDDEN_DIM_NUM = 5

    # checkpoint = torch.load('./checkpoint/DQN_model_breakout_450.0.pth')

    dqn1 = DQN_CNN_Super(args)
    dqn2 = DQN_CNN_Super(args)
    # double_dqn = DQN_CNN(args, NAME="DDQN")
    # double_dqn.main_net.load_state_dict(dqn.main_net.state_dict())
    # double_dqn.target_net.load_state_dict(dqn.target_net.state_dict())
    # double_dqn.main_net.load_state_dict(checkpoint['main_net_state_dict'])
    # double_dqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])

    dqn2.main_net.load_state_dict(dqn1.main_net.state_dict())
    dqn2.target_net.load_state_dict(dqn1.target_net.state_dict())

    agent1 = Agent_Experiment(args, dqn1)
    agent1.train_breakout(use_super=True)

    agent2 = Agent_Experiment(args, dqn2)
    agent2.train_breakout(order=2, use_super=False)

    # torch.save({"main_net_state_dict": dqn.main_net.state_dict(),
    #             "target_net_state_dict": dqn.target_net.state_dict()}, "checkpoint/dqn_model_breakout_final.pth")
    # torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
    #             "target_net_state_dict": double_dqn.target_net.state_dict()},
    #            "checkpoint/DDQN_model_breakout_final.pth")
    plt.show()


def Seaquest_experiment():
    args = SetupArgs().get_args()

    args.num_episodes = 50000
    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 18
    args.HIDDEN_DIM_NUM = 5

    # checkpoint = torch.load('./checkpoint/DQN_model_breakout_450.0.pth')

    dqn = DQN_CNN_Super(args)
    dqnCopy = copy.deepcopy(dqn)
    double_dqn = DQN_CNN(args, NAME="DDQN")
    double_dqn.main_net.load_state_dict(dqn.main_net.state_dict())
    double_dqn.target_net.load_state_dict(dqn.target_net.state_dict())
    # double_dqn.main_net.load_state_dict(checkpoint['main_net_state_dict'])
    # double_dqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent1 = Agent_Experiment(args, dqn)
    agent1.train_Seaquest(use_super=True)

    agent2 = Agent(args, dqnCopy)
    agent2.train_Seaquest(RND=False, order=2)

    torch.save({"main_net_state_dict": dqn.main_net.state_dict(),
                "target_net_state_dict": dqn.target_net.state_dict()}, "checkpoint/dqn_model_Seaquest_final.pth")
    torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
                "target_net_state_dict": double_dqn.target_net.state_dict()},
               "checkpoint/DDQN_model_Seaquest_final.pth")
    plt.show()


def Pong_experiment():
    args = SetupArgs().get_args()

    args.num_episodes = 10000
    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 6
    args.HIDDEN_DIM_NUM = 5

    # checkpoint = torch.load('./checkpoint/DQN_model_breakout_450.0.pth')

    dqn = DQN_CNN_Super(args)
    double_dqn = DQN_CNN(args, NAME="DDQN")
    double_dqn.main_net.load_state_dict(dqn.main_net.state_dict())
    double_dqn.target_net.load_state_dict(dqn.target_net.state_dict())
    # double_dqn.main_net.load_state_dict(checkpoint['main_net_state_dict'])
    # double_dqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent1 = Agent_Experiment(args, dqn)
    agent1.train_Pong(use_super=True)

    # agent2 = Agent(args, double_dqn)
    # agent2.train_breakout(RND=True, order=2)

    torch.save({"main_net_state_dict": dqn.main_net.state_dict(),
                "target_net_state_dict": dqn.target_net.state_dict()}, "checkpoint/dqn_model_Pong_final.pth")
    torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
                "target_net_state_dict": double_dqn.target_net.state_dict()},
               "checkpoint/DDQN_model_Pong_final.pth")
    plt.show()


"""
--------------------------以下是实现了优先经验回放以及调用了库中实现DQN算法的内容
"""


def breakout_experiment_lib():
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 4
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 20,
        'width_end': 210,
        'height_start': 0,
        'height_end': 160
    }
    args.rnd = {
        'use_rnd': True,
        'rnd_weight': 0.01,
        'rnd_weight_decay': 1,
    }
    args.env_name = "ALE/Breakout-v5"
    args.reward_cut = 1

    experiment.create(name="ppo")

    DQN_configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': False,
        'rnd': {
            'use_rnd': True,
            'rnd_weight': 0.01,
            'rnd_weight_decay': 1,
        },
        'test': {
            'use_test': False,
            'test_model': None,
        },
        'algorithm_name': "Dueling DQN"
    }

    PPO_configs = {
        'updates': 10000,

        'epochs': IntDynamicHyperParam(8),

        'n_workers': 8,

        'worker_steps': 128,

        'batches': 4,

        'value_loss_coef': FloatDynamicHyperParam(0.5),

        'entropy_bonus_coef': FloatDynamicHyperParam(0.01),

        'clip_range': FloatDynamicHyperParam(0.1),

        'learning_rate': FloatDynamicHyperParam(1e-3, (0, 1e-3)),

        'args': args,

        'test': {
            'use_test': False,
            'test_model': None,
        },
    }

    experiment.configs(PPO_configs)

    m = PPOTrainer(**PPO_configs)

    with experiment.start():
        m.run_training_loop()

    m.destroy()


def RoadRunner_experiment_lib():
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 18
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 105,
        'width_end': 180,
        'height_start': 0,
        'height_end': 160
    }
    args.reward_cut = 0.01

    experiment.create(name="dqn")

    configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': True,
        'test': False,
        'algorithm_name': "DQN"
    }
    configs_copy = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': False,
        'test': False,
        'algorithm_name': "DQN"
    }

    experiment.configs(configs)

    m = DQN_Super_Trainer(**configs)
    n = DQN_Super_Trainer(**configs_copy)

    with experiment.start():
        m.run_training_loop()
        n.run_training_loop()

    m.destroy()
    n.destroy()


def Seaquest_experiment_lib():
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 18
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 35,
        'width_end': 188,
        'height_start': 8,
        'height_end': 160
    }
    args.reward_cut = 0.01

    experiment.create(name="dqn")

    configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': True,
        'test': False,
        'algorithm_name': "DQN"
    }

    experiment.configs(configs)

    m = DQN_Super_Trainer(**configs)

    with experiment.start():
        m.run_training_loop()

    m.destroy()


def Freeway_experiment_lib():
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 3
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 12,
        'width_end': 196,
        'height_start': 0,
        'height_end': 160
    }
    args.reward_cut = 1

    experiment.create(name="dqn")

    configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': True,
        'test': False,
        'algorithm_name': "DQN"
    }

    experiment.configs(configs)

    m = DQN_Super_Trainer(**configs)

    with experiment.start():
        m.run_training_loop()

    m.destroy()


def Tetris_experiment_lib():  # 40000回合训练无训练效果，考虑该环境不能使用DQN进行训练。或者需要更多轮次。
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 5
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 25,
        'width_end': 195,
        'height_start': 22,
        'height_end': 64
    }
    args.reward_cut = 1

    experiment.create(name="dqn")

    configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': True,
        'test': False,
        'algorithm_name': "DQN"
    }

    experiment.configs(configs)

    m = DQN_Super_Trainer(**configs)

    with experiment.start():
        m.run_training_loop()

    m.destroy()


def Pong_experiment_lib():
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 6
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 30,
        'width_end': -12,
        'height_start': 5,
        'height_end': -14
    }
    args.reward_cut = 1

    experiment.create(name="dqn")

    configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': True,
        'test': False,
        'algorithm_name': "DQN"
    }

    experiment.configs(configs)

    m = DQN_Super_Trainer(**configs)

    with experiment.start():
        m.run_training_loop()

    m.destroy()


def Bowling_experiment_lib():
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 6
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 100,
        'width_end': 175,
        'height_start': 0,
        'height_end': 160
    }
    args.reward_cut = 0.01

    experiment.create(name="dqn")

    configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 8,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': True,
        'test': False,
        'algorithm_name': "DQN"
    }

    experiment.configs(configs)

    m = DQN_Super_Trainer(**configs)

    with experiment.start():
        m.run_training_loop()

    m.destroy()


def Montezuma_revenge_experiment_lib():
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 18
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 20,
        'width_end': 210,
        'height_start': 0,
        'height_end': 160
    }
    args.rnd = {
            'use_rnd': True,
            'rnd_weight': 0.01,
            'rnd_weight_decay': 1,
    }
    args.env_name = "ALE/MontezumaRevenge-v5"
    args.reward_cut = 1

    experiment.create(name="ppo")

    DQN_configs = {
        'updates': 1000000,
        'epochs': 8,
        'n_workers': 1,
        'worker_steps': 4,
        'mini_batch_size': 32,
        'update_target_model': 250,
        'learning_rate': FloatDynamicHyperParam(1e-4, (0, 1e-3)),
        'args': args,
        'use_super': False,
        'rnd': {
            'use_rnd': True,
            'rnd_weight': 0.01,
            'rnd_weight_decay': 1,
        },
        'test': {
            'use_test': True,
            'test_model': './checkpoint/dqn_MontezumaRevenge-v5_23_07_09_14_.pth',
        },
        'algorithm_name': "Dueling DQN"
    }

    PPO_configs = {
        'updates': 10000,

        'epochs': IntDynamicHyperParam(8),

        'n_workers': 8,

        'worker_steps': 128,

        'batches': 4,

        'value_loss_coef': FloatDynamicHyperParam(0.5),

        'entropy_bonus_coef': FloatDynamicHyperParam(0.01),

        'clip_range': FloatDynamicHyperParam(0.1),

        'learning_rate': FloatDynamicHyperParam(1e-3, (0, 1e-3)),

        'args': args,

        'test': {
            'use_test': False,
            'test_model': None,
        },
    }

    experiment.configs(PPO_configs)

    m = PPOTrainer(**PPO_configs)

    with experiment.start():
        m.run_training_loop()

    m.destroy()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    Montezuma_revenge_experiment_lib()

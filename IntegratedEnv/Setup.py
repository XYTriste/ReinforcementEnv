# -*- coding = utf-8 -*-
# @time: 5/26/2023 8:52 PM
# Author: Yu Xia
# @File: Setup.py.py
# @software: PyCharm
import copy


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

    args.num_episodes = 50000
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

    args.num_episodes = 10000
    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 4
    args.HIDDEN_DIM_NUM = 5

    # checkpoint = torch.load('./checkpoint/DQN_model_breakout_450.0.pth')

    dqn = DQN_CNN_Super(args)
    double_dqn = DQN_CNN(args, NAME="DDQN")
    double_dqn.main_net.load_state_dict(dqn.main_net.state_dict())
    double_dqn.target_net.load_state_dict(dqn.target_net.state_dict())
    # double_dqn.main_net.load_state_dict(checkpoint['main_net_state_dict'])
    # double_dqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent1 = Agent_Experiment(args, dqn)
    agent1.train_breakout(use_super=True)

    # agent2 = Agent(args, double_dqn)
    # agent2.train_breakout(RND=True, order=2)

    torch.save({"main_net_state_dict": dqn.main_net.state_dict(),
                "target_net_state_dict": dqn.target_net.state_dict()}, "checkpoint/dqn_model_breakout_final.pth")
    torch.save({"main_net_state_dict": double_dqn.main_net.state_dict(),
                "target_net_state_dict": double_dqn.target_net.state_dict()},
               "checkpoint/DDQN_model_breakout_final.pth")
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

def Pong_experiment():
    args = SetupArgs().get_args()

    args.num_episodes = 50000
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


if __name__ == "__main__":
    Seaquest_experiment()

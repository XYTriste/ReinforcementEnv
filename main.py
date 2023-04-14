from typing import Optional, Union, List, Tuple

import gym
from gym.core import RenderFrame, ActType, ObsType
from matplotlib import pyplot as plt

from GridWind.Agent import *
from GridWind.WindGridTest import get_average_data

env = WindGridEnv()
env.reset()
agent = WindGridAgent(env)
flg, ax = plt.subplots()

step_list_sarsa_1, _ = agent.sarsa_algorithm(setflag=True, rounds=10000, isTrain=False, gamma=0.91, epsilon=0.538,
                                                 alpha=0.156)
step_list_sarsa_lambda_1, _ = agent.sarsa_lambda_algorithm(setflag=True, rounds=10000, isTrain=False, gamma=0.86,
                                                               epsilon=0.320, alpha=0.0241)
average_list_s = get_average_data(step_list_sarsa_1, 100)
average_list_ss = get_average_data(step_list_sarsa_lambda_1, 100)
episode_list = [i for i in range(0, 10000, 100)]
ax.plot(episode_list, average_list_s, label="sarsa")
ax.legend()
plt.savefig("./testsarsa.png")
plt.close(flg)

flg, ax = plt.subplots()
ax.plot(episode_list, average_list_ss, label="sarsa lambda")
ax.legend()
plt.savefig("./test sarsa lambda.png")

plt.close(flg)
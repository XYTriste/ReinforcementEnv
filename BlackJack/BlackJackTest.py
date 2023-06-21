import numpy as np

import gymnasium
from BlackJackEnv_v1 import *
from BlackJackAgent import *

env = gymnasium.make("Blackjack-v1", render_mode="rgb_array")
agent = BlackJackAgent(env)
# rounds = [100000, 1000000, 10000000]
rounds = [30000000]
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
player_prob = [0, 0.005917159763313609, 0.011834319526627219, 0.01775147928994083, 0.023668639053254437,
               0.029585798816568046,
               0.03550295857988166, 0.04142011834319527, 0.047337278106508875, 0.05325443786982249, 0.09467455621301775,
               0.08875739644970414, 0.08284023668639054, 0.07692307692307693, 0.07100591715976332, 0.0650887573964497,
               0.05917159763313609, 0.05325443786982249, 0.047337278106508875, 0.09467455621301775]
dealer_prob = [0] + [deck.count(i) / len(deck) for i in range(1, 11)]
player_prob = np.array(player_prob)
dealer_prob = np.array(dealer_prob)
prob = {}
for i in range(2, 21):
    for j in range(1, 11):
        prob[i, j] = player_prob[i - 1] * dealer_prob[j]
# print(sum(prob.values()))
print("Q learning:")
for round in rounds:
    agent.Q_learning_algorithm(rounds=round, epsilon=0.01)
    # print("Training complete")
    #info = agent.play_with_dealer()
    agent.calc_state_value()
    expectation_by_train = np.dot(np.array(list(prob.values())), np.array(
        list(agent.state_value_function.values())[:190]))
    #info = agent.play_with_dealer(use_best_policy=True)
    agent.Q_learning_algorithm(rounds=round, epsilon=0.01, use_best_policy=True)
    agent.calc_state_value()
    expectation_by_calc = np.dot(np.array(list(prob.values())), np.array(
        list(agent.state_value_function.values())[:190]))
    print("使用epsilon greedy策略训练{}回合， 起始状态价值的期望:{:6f}".format(round, expectation_by_train))
    print("使用最优策略训练{}回合， 起始状态价值的期望:{:6f}".format(round, expectation_by_calc))
    print("绝对误差为:{:4f},  相对误差为:{:4f}, 相对性能提升约:{:4f}%".format(expectation_by_calc - expectation_by_train,
                                                                             (expectation_by_calc - expectation_by_train) / expectation_by_calc,
                                                                             (expectation_by_calc - expectation_by_train) / expectation_by_calc * 100))
    # print("Player win rate: {:.2f}%   Dealer win rate:{:.2f}%   Not lose rate:{:.2f}%"
    #       .format(info[0] / round * 100, info[1] / round * 100, (round - info[1]) / round * 100))
    # for i in range(2, 21):
    #     for j in range(1, 11):
    #         for k in range(2):
    #             print("({}, {}, {}) = {:5f}".format(i, j, k, agent.action_value_function[i, j, False][k]), end="  ")
    #         print()

    # for i in range(2, 22):
    #     for j in range(1, 11):
    #         print("({}, {}) = {}".format(i, j, agent.state_value_function[i, j]))



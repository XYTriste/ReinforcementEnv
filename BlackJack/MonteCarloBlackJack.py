from BlackJackEnv_v1 import *
import numpy as np
from BlackJackUtils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

env = gym.make("Blackjack-v1", render_mode="rgb_array")

V = np.zeros((32, 12))  # 状态价值
V_N = np.zeros((32, 12))  # 状态出现次数
Q = np.zeros((32, 11, 2))  # 行为价值
# Q = np.random.rand((32, 2)) * 1E-5
Q_N = np.zeros((32, 2))  # 行为出现次数
gamma = 0.99
epsilon = 0.1
record = {}

win_count = {}
lose_count = {}


def initialization_training_data():
    global V
    global V_N
    global Q
    global Q_N
    global epsilon
    global win_count
    global lose_count

    V = np.zeros((32, 12))  # 状态价值
    V_N = np.zeros((32, 12))  # 状态出现次数
    Q = np.zeros((32, 11, 2))  # 行为价值
    Q_N = np.zeros((32, 11, 2))  # 行为出现次数
    epsilon = 1

    for i in range(1, 32):
        win_count[i] = 0
        lose_count[i] = 0


def greedy_policy(state):
    return np.argmax(Q[state])


def epsilon_greedy_policy(player_state, dealer_state, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        Q_state = Q[player_state, dealer_state]
        return np.argmax(Q_state)


def greedy_policy(player_state, dealer_state):
    return np.argmax(Q[player_state, dealer_state])


def player_policy(player_state, dealer_state):
    global win_count
    global lose_count
    curr_win_rate = win_count[player_state] / (win_count[player_state] + lose_count[player_state])
    if curr_win_rate < 0.5:
        return 1
    else:
        return np.argmax(Q[player_state, dealer_state])


def easy_policy(player_state):
    if player_state < 15:
        return 1
    else:
        return 0


def monte_carlo():
    global epsilon
    initialization_training_data()
    player_win = 0
    dealer_win = 0
    state_count = {}

    train_rounds = 0
    rounds = 150000
    ALPHA = 0.002
    average_step = (1 - ALPHA) / rounds

    visited_state = set()  # 记录状态是否出现过

    # while train_rounds < 30000000 or (train_rounds < 30000000 and player_win / train_rounds < 44.0):
    for i in range(rounds):

        # percent = rounds / 20
        # if i > 0 and i % percent == 0:
        #     pass
            # print("训练完了 {} 回合, 玩家赢了: {} 回合   庄家赢了: {} 回合".format(i, player_win, dealer_win))
            # process_bar(rounds, i)
            # play_with_dealer(10000, i)

        if train_rounds % 100000 == 0 and train_rounds > 0:
            print("train rounds:{}   player win rate:{}".format(train_rounds, player_win / train_rounds))
        obs, _ = env.reset()
        player_state, dealer_state, usable_ace = obs

        episode = []

        done = False

        while not done:
            action = epsilon_greedy_policy(player_state, dealer_state, epsilon)
            observation, reward, done, _, _ = env.step(action)

            episode.append((player_state, action, reward, dealer_state))
            player_state = observation[0]

            if done:
                if player_state in state_count.keys():
                    state_count[player_state] += 1
                else:
                    state_count[player_state] = 1
        if reward > 0:
            player_win += 1
            win_count[player_state] += 1
        elif reward < 0:
            lose_count[player_state] += 1

        # if epsilon > 0.001:
        #     epsilon *= 0.99

        # if episode[-1][-1] >= 0:
        #     player_win += 1
        # elif episode[-1][-1] < 0:
        #     dealer_win += 1
        train_rounds += 1
        # G = 0
        # returns = []
        # for t in range(len(episode) - 1, -1, -1):
        #     G = episode[t][2] + gamma * G
        #     returns = [G] + returns
        #
        # visited_set = set()
        # index = 0
        #
        # update_alpha = ALPHA
        # ALPHA += average_step
        # for state, action, reward in episode:
        #     if (state, action) not in visited_set:
        #         visited_set.add((state, action))
        #         V_N[state, dealer_state] += 1
        #         alpha = 1.0 / V_N[state, dealer_state]
        #         V[state, dealer_state] += alpha * (returns[index] - V[state, dealer_state])
        #
        #         # Q_N[state, action] += 1
        #         # alpha = 1.0 / Q_N[state, action]
        #         # Q[state, action] += update_alpha * (returns[index] - Q[state, action])
        #
        #         if state in visited_state:
        #             Q[state, action] += (1 - ALPHA) * (returns[index] - Q[state, action])
        #         else:
        #             visited_state.add(state)
        #             Q[state][action] = ALPHA * returns[index]
        #     index += 1
        returns = 0
        for state, action, reward, dealer_state in reversed(episode):
            returns = gamma * returns + reward
            # if (state, dealer_state) in visited_state:
            Q[state, dealer_state, action] = (1 - ALPHA) * Q[state, dealer_state, action] + ALPHA * returns
            # else:
            #     # print("player:{}  dealer:{}".format(state, dealer_state))
            #     visited_state.add((state, dealer_state))
            #     Q[state, dealer_state, :] = 0
            #     Q[state, dealer_state, action] = ALPHA * returns


def play_with_dealer(rounds, trained_rounds):
    """
    以下是玩家与庄家对抗时的信息
    """
    player_win = 0
    dealer_win = 0
    draw = 0

    weight = 0.7
    for i in range(rounds):
        done = False

        obs, _ = env.reset()
        player_state, dealer_state, _ = obs


        # print(sum(env.player_win_rate.values()))

        while not done:
            action = greedy_policy(player_state, dealer_state)
            observation, reward, done, _, _ = env.step(action)
            player_state = observation[0]
            dealer_state = observation[1]

        if reward > 0:
            player_win += 1
        elif reward == 0:
            draw += 1
        else:
            dealer_win += 1
    print_winning_probability(rounds, trained_rounds, player_win, dealer_win, draw)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # 减去最大值，用于数值稳定性
    return e_x / np.sum(e_x, axis=0)


if __name__ == '__main__':
    monte_carlo()
    play_with_dealer(10000, 10000)

    for i in range(2, 32):
        for j in range(2, 11):
            print("玩家点数为{} 庄家点数为 {} 时的最佳动作是: {}".format(i, j, "抽牌" if np.argmax(Q[i, j]) else "不抽牌"))

    best_actions = np.zeros((32, 11))

    # 根据最佳行为设置相应位置为0或1
    for i in range(32):
        for j in range(11):
            best_actions[i, j] = np.argmax(Q[i, j])

    # best_actions = best_actions[10:22, :]
    # 绘制热力图
    plt.imshow(best_actions[:, 1:], cmap='coolwarm', aspect='auto')

    # 添加坐标轴标签
    plt.xlabel("Dealer state")
    plt.ylabel("Player state")

    # 设置坐标轴刻度
    plt.xticks(range(1, 11))
    plt.yticks(range(1, 32))

    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["stick", "hit"])

    plt.grid(True)

    # 显示图形
    plt.show()
from BlackJackEnv_v0 import *
import numpy as np
from BlackJackUtils import *

env = BlackJackEnvironment()

V = np.zeros((32, 12))  # 状态价值
V_N = np.zeros((32, 12))  # 状态出现次数
Q = np.zeros((32, 2))  # 行为价值
# Q = np.random.rand((32, 2)) * 1E-5
Q_N = np.zeros((32, 2))  # 行为出现次数
gamma = 0.95
epsilon = 1
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
    Q = np.zeros((32, 2))  # 行为价值
    Q_N = np.zeros((32, 2))  # 行为出现次数
    epsilon = 1

    for i in range(1, 32):
        win_count[i] = 0
        lose_count[i] = 0


def greedy_policy(state):
    return np.argmax(Q[state])


def epsilon_greedy_policy(player_state, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        Q_state = Q[player_state]
        return np.argmax(Q_state)


def player_policy(player_state):
    global win_count
    global lose_count
    curr_win_rate = win_count[player_state] / (win_count[player_state] + lose_count[player_state])
    if curr_win_rate < 0.5:
        return 1
    else:
        return np.argmax(Q[player_state])


def monte_carlo():
    global epsilon
    initialization_training_data()
    player_win = 0
    dealer_win = 0

    rounds = 500000
    ALPHA = 0.002
    average_step = (1 - ALPHA) / rounds

    visited_state = set()  # 记录状态是否出现过

    for i in range(rounds):

        percent = rounds / 20
        if i > 0 and i % percent == 0:
            print("训练完了 {} 回合, 玩家赢了: {} 回合   庄家赢了: {} 回合".format(i, player_win, dealer_win))
            process_bar(rounds, i)
            play_with_dealer(10000, i)

        obs, _ = env.reset()
        player_state, dealer_state, usable_ace = obs

        episode = []

        done = False

        while not done:
            action = epsilon_greedy_policy(player_state, epsilon)
            observation, reward, done, _, _ = env.step(action)

            episode.append((player_state, action, reward))
            player_state = observation[0]

            if done:
                if reward > 0:
                    win_count[player_state] += 1
                else:
                    lose_count[player_state] += 1

        if epsilon > 0.1:
            epsilon *= 0.99

        if episode[-1][-1] >= 0:
            player_win += 1
        elif episode[-1][-1] < 0:
            dealer_win += 1
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
        for state, action, reward in reversed(episode):
            returns = gamma * returns + reward
            if state in visited_state:
                Q[state, action] = (1 - ALPHA) * Q[state, action] + ALPHA * returns
            else:
                visited_state.add(state)
                Q[state, :] = 0
                Q[state, action] = ALPHA * returns


def play_with_dealer(rounds, trained_rounds):
    """
    以下是玩家与庄家对抗时的信息
    """
    player_win = 0
    dealer_win = 0
    draw = 0
    for i in range(rounds):
        done = False

        obs, _ = env.reset()
        player_state, dealer_state, _ = obs
        while not done:
            action = epsilon_greedy_policy(player_state)
            observation, reward, done, _, _ = env.step(action)
            player_state = observation[0]

        if reward > 0:
            player_win += 1
        elif reward == 0:
            draw += 1
        else:
            dealer_win += 1
    print_winning_probability(rounds, trained_rounds, player_win, dealer_win, draw)


if __name__ == '__main__':
    monte_carlo()
    play_with_dealer(10000, 50000)
    # for i in range(4, 22):
    #     for j in range(2):
    #         print("状态 {} 时, 行为 {} 的行为价值为:{}".format(i, "不抽牌" if j == 0 else "抽牌", Q[i, j]))
    #     print("{} 更好".format("抽牌" if Q[i, 1] > Q[i, 0] else "不抽牌"))
    #     print()

    for i in range(1, 32):
        print("Player State:{}   win rate:{}   lose rate:{}".format(i, win_count[i] / (win_count[i] + lose_count[i]),
                                                                    lose_count[i] / (win_count[i] + lose_count[i])))

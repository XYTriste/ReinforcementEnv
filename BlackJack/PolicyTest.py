# -*- coding = utf-8 -*-
# @time: 6/17/2023 8:58 PM
# Author: Yu Xia
# @File: PolicyTest.py
# @software: PyCharm
import itertools
import gymnasium


def basic_policy(state):
    player_state = state[0]
    if player_state < 15:
        return 1
    else:
        return 0


def policy_by_sarsa(rounds, state):
    player_state, dealer_state = state[0], state[1]
    if rounds == 100000:
        if player_state == 2 or player_state == 3:
            return 1
        elif player_state >= 4 and player_state <= 11:
            return 1
        elif player_state == 12:
            if dealer_state == 3 or dealer_state == 6:
                return 0
            else:
                return 1
        elif player_state == 13:
            if 3 <= dealer_state <= 6:
                return 0
            else:
                return 1
        elif player_state == 14:
            if dealer_state == 2 or 4 <= dealer_state <= 6 or dealer_state == 8:
                return 0
            else:
                return 1
        elif player_state == 15:
            if 2 <= dealer_state <= 6:
                return 0
            else:
                return 1
        elif player_state == 16:
            if 2 <= dealer_state <= 8:
                return 0
            else:
                return 1
        else:
            return 0
    elif rounds == 1000000:
        if player_state == 2 or player_state == 3:
            return 1
        elif 4 <= player_state <= 9 or player_state == 11:
            return 1
        elif player_state == 10:
            if dealer_state == 2:
                return 0
            else:
                return 1
        elif player_state == 12:
            if 4 <= dealer_state <= 6:
                return 0
            else:
                return 1
        elif player_state == 13:
            if 4 <= dealer_state <= 7:
                return 0
            else:
                return 1
        elif player_state == 14:
            if 3 <= dealer_state <= 5:
                return 0
            else:
                return 1
        elif player_state == 15:
            if dealer_state == 2 or 4 <= dealer_state <= 6:
                return 0
            else:
                return 1
        elif player_state == 16:
            if 2 <= dealer_state <= 7:
                return 0
            else:
                return 1
        elif player_state == 17:
            if 2 <= dealer_state <= 8:
                return 0
            else:
                return 1
        else:
            return 0
    elif rounds == 10000000:
        if player_state == 2 or player_state == 3:
            return 1
        elif 4 <= player_state <= 12:
            if player_state == 12 and dealer_state == 3:
                return 0
            else:
                return 1
        elif player_state == 13:
            if dealer_state == 2 or 5 <= dealer_state <= 6:
                return 0
            else:
                return 1
        elif player_state == 14:
            if 2 <= dealer_state <= 6 or dealer_state == 8 or dealer_state == 10:
                return 0
            else:
                return 1
        elif player_state == 15:
            if 2 <= dealer_state <= 3 or dealer_state == 6:
                return 0
            else:
                return 1
        elif player_state == 16:
            if 2 <= dealer_state <= 6 or dealer_state == 8:
                return 0
            else:
                return 1
        elif player_state == 17:
            if dealer_state == 1 or dealer_state == 8:
                return 1
            else:
                return 0
        else:
            return 0

def best_policy(round, state):
    player_state, dealer_state = state[0], state[1]
    if player_state <= 11:
        return 1
    elif player_state == 12:
        if 4 <= dealer_state <= 6:
            return 0
        else:
            return 1
    elif player_state >= 13 and player_state <= 16:
        if 2 <= dealer_state <= 6:
            return 0
        else:
            return 1
    else:
        return 0

if __name__ == '__main__':
    #给定数组
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    # 所有可能的组合
    combinations = list(itertools.product(arr, repeat=2))

    # 统计每种和的出现次数
    sum_counts = {}
    for comb in combinations:
        total_sum = sum(comb)
        sum_counts[total_sum] = sum_counts.get(total_sum, 0) + 1

    # 计算每种和的概率
    total_combinations = len(combinations)
    probabilities = {}
    for sum_val, count in sum_counts.items():
        probabilities[sum_val] = count / total_combinations

    # 打印结果
    for sum_val, prob in probabilities.items():
        print(f"和为 {sum_val} 的概率为 {prob:.4f}")
    print(probabilities.values())
    # env = gymnasium.make("Blackjack-v1", render_mode="rgb_array")
    # rounds = [100000, 1000000, 10000000]
    # for round in rounds:
    #     principal = 2000
    #     for i in range(2000):
    #         state, _ = env.reset()
    #         done = False
    #         while not done:
    #             action = policy_by_sarsa(round, state)
    #             state, reward, done, _, _ = env.step(action)
    #
    #         if reward > 0:
    #             principal += 1
    #         elif reward < 0:
    #             principal -= 1
    #
    #     print("{} 训练回合的策略在游戏2000回合后的本金是:{}".format(round, principal))
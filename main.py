import random


def calculate_bust_probability():
    """计算在每个点数时爆牌的概率"""
    bust_probability = [0] * 32  # 初始化每个点数时爆牌的概率

    num_simulations = 100000  # 模拟游戏回合数

    for _ in range(num_simulations):
        hand_value = 0
        while hand_value < 21:
            card_value = random.randint(1, 10)  # 随机抽取一张牌，牌值范围为1到10
            hand_value += card_value
            if hand_value > 21:
                bust_probability[hand_value] += 1
                break

    bust_probability = [count / num_simulations for count in bust_probability]
    return bust_probability


bust_probability = calculate_bust_probability()

for points, probability in enumerate(bust_probability):
    print(f"点数为 {points} 时爆牌的概率：{probability:.4f}")

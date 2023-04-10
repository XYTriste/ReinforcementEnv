record = {}

def process_bar(rounds, i):
    print("Training {:.4f}%".format((i / rounds) * 100))


def print_winning_probability(rounds, trained_rounds, player_win, dealer_win, draw):
    player_winning_rate = (player_win / rounds) * 100
    dealer_winning_rate = (dealer_win / rounds) * 100
    not_lose_rate = (player_win + draw) / rounds * 100
    print("After trained {} rounds. Player win:{}   Dealer win:{}.".format(trained_rounds, player_win, dealer_win))
    print("Player win rate:{:.2f}%.  not lose rate:{:.2f}%.  Dealer win rate:{:.2f}%".format(player_winning_rate,
                                                                                             not_lose_rate,
                                                                                             dealer_winning_rate))
    record_winning_rate(trained_rounds, player_winning_rate, dealer_winning_rate, not_lose_rate)


def record_winning_rate(trained_rounds, player_winning_rate, dealer_winning_rate, not_lose_rate):
    global record
    record[trained_rounds] = (player_winning_rate, dealer_winning_rate, not_lose_rate)


def show_value(V, Q):
    for i in range(4, 22):
        print("状态 {} 时状态价值为: {}".format(i, V[i]))
        max_action = Q[i, 0, 0]
        for j in range(2):
            print("该状态下，动作 {} 的价值为: {}".format("stick" if j == 0 else "hit", Q[i, j]))
            if Q[i, j] > max_action:
                max_action = Q[i, j]
        print()

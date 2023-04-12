import matplotlib.pyplot as plt
from Agent import *

env = WindGridEnv()
env.reset()
agent = WindGridAgent(env)


def calculateGroup(step_list, group_size):
    averages_step = []
    average_episode = []
    episode_list = [i for i in range(10000)]

    for i in range(0, len(step_list), group_size):
        group = step_list[i: i + group_size]
        avg = sum(group) / len(group)

        group_episode = episode_list[i: i + group_size]
        avg_episode = sum(group_episode) / len(group_episode)

        averages_step.append(avg)
        average_episode.append(avg_episode)

    return average_episode, averages_step


def calculate_sarsa():
    r_0 = (0, -1, 1)
    r_1 = (-1, -1, 1)

    rounds = 1000

    episode_list = [i for i in range(rounds)]
    step_list_sarsa_0 = agent.sarsa_algorithm(rounds, r_0)
    step_list_sarsa_1 = agent.sarsa_algorithm(rounds, r_1)
    step_list_sarsa_lambda_0 = agent.sarsa_lambda_algorithm(rounds, r_0)
    step_list_sarsa_lambda_1 = agent.sarsa_lambda_algorithm(rounds, r_1)

    flg, ax = plt.subplots()

    average_episode, averages_step_sarsa_0 = calculateGroup(step_list_sarsa_0, 100)
    _, averages_step_sarsa_1 = calculateGroup(step_list_sarsa_1, 100)
    ax.plot(average_episode, averages_step_sarsa_0, label="(0, -1, 1)")
    ax.plot(average_episode, averages_step_sarsa_1, label="(-1, -1, 1)")

    ax.legend()
    plt.savefig("./images/difference.png")
    plt.show()


if __name__ == '__main__':
    r_1 = (0, -1, 1)

    rounds = 200000

    step_list_Q_learning, _ = agent.Q_learning_algorithm(setflag=True, alpha=0.1873, epsilon=0.43, gamma=0.89, lambda_=0.7834)
    # step_list_sarsa_1, _ = agent.sarsa_algorithm(setflag=False, alpha=0.1671, epsilon=0.3696, gamma=0.8379, rounds=10000)

    flg, ax = plt.subplots()
    average_episode, averages_step_sarsa_1 = calculateGroup(step_list_Q_learning, 100)
    ax.plot(average_episode, averages_step_sarsa_1, label="(-1, -1, 1)")
    ax.legend()
    plt.show()

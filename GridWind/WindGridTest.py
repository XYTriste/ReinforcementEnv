from GridWind.Agent import *
import matplotlib.pyplot as plt


def calculateGroup(step_list, group_size):
    global episode_list

    averages_step = []
    average_episode = []

    for i in range(0, len(step_list), group_size):
        group = step_list[i: i + group_size]
        avg = sum(group) / len(group)

        group_episode = episode_list[i: i + group_size]
        avg_episode = sum(group_episode) / len(group_episode)

        averages_step.append(avg)
        average_episode.append(avg_episode)

    return average_episode, averages_step


r_0 = (0, -1, 1)
r_1 = (-1, -1, 1)

rounds = 1000

env = WindGridEnv()
env.reset()
agent = WindGridAgent(env)

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
plt.show()
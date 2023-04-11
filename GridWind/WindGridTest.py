from GridWind.Agent import *
import matplotlib.pyplot as plt

r_0 = (0, -1, 1)
r_1 = (-1, -1, 1)

rounds = 10000

env = WindGridEnv()
env.reset()
agent = WindGridAgent(env)

episode_list = [i for i in range(rounds)]
step_list_1 = agent.sarsa_algorithm(rounds, r_1)

group_size = 100
averages = []
average_episode = []
for i in range(0, len(step_list_1), group_size):
    group = step_list_1[i: i + group_size]
    avg = sum(group) / len(group)

    group_episode = episode_list[i: i + group_size]
    avg_episode = sum(group_episode) / len(group_episode)

    averages.append(avg)
    average_episode.append(avg_episode)

plt.plot(average_episode, averages)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Training Process')
plt.show()

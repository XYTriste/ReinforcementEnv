import matplotlib.pyplot as plt
from Agent import *

env = WindGridEnv()
env.reset()
agent = WindGridAgent(env)
flg, ax = plt.subplots()

episode_list = [i * 100 for i in range(100)]


def calculateGroup(step_list, group_size):
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


def calculate_sarsa():
    r_0 = (0, -1, 1)
    r_1 = (-1, -1, 1)

    rounds = 1000

    episode_list = [i for i in range(rounds)]
    step_list_sarsa_0 = agent.sarsa_algorithm(rounds, r_0)
    step_list_sarsa_1 = agent.sarsa_algorithm(rounds, r_1)
    step_list_sarsa_lambda_0 = agent.sarsa_lambda_algorithm(rounds, r_0)
    step_list_sarsa_lambda_1 = agent.sarsa_lambda_algorithm(rounds, r_1)

    average_episode, averages_step_sarsa_0 = calculateGroup(step_list_sarsa_0, 100)
    _, averages_step_sarsa_1 = calculateGroup(step_list_sarsa_1, 100)
    ax.plot(average_episode, averages_step_sarsa_0, label="(0, -1, 1)")
    ax.plot(average_episode, averages_step_sarsa_1, label="(-1, -1, 1)")

    ax.legend()
    plt.savefig("./images/difference.png")
    plt.show()


def training_model():
    step_list_sarsa_0, _ = agent.sarsa_algorithm(setflag=False, rounds=10000, isTrain=False)
    step_list_sarsa_1, _ = agent.sarsa_algorithm(setflag=True, rounds=10000, isTrain=False)
    step_list_sarsa_lambda_0, _ = agent.sarsa_lambda_algorithm(setflag=False, rounds=10000, isTrain=False)
    step_list_sarsa_lambda_1, _ = agent.sarsa_lambda_algorithm(setflag=True, rounds=10000, isTrain=False)
    step_list_Q_learning_0, _ = agent.Q_learning_algorithm(setflag=False, rounds=10000, isTrain=False)
    step_list_Q_learning_1, _ = agent.Q_learning_algorithm(setflag=True, rounds=10000, isTrain=False)

    return step_list_sarsa_0, \
        step_list_sarsa_1, \
        step_list_sarsa_lambda_0, \
        step_list_sarsa_lambda_1, \
        step_list_Q_learning_0, \
        step_list_Q_learning_1


def get_average_data(step_list, group_size):
    averages_step = []

    for i in range(0, len(step_list), group_size):
        group = step_list[i: i + group_size]
        avg = sum(group) / len(group)

        averages_step.append(avg)

    return averages_step


def paint_img(x, y, imageName, label):
    global ax
    path = "./images/" + imageName + ".png"

    if type(y) is list:
        for i in range(0, len(y)):
            ax.plot(x, y[i], label=label[i])
    else:
        ax.plot(x, y, label=label)
    ax.legend()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    all_step_list = training_model()
    average_list = []
    for i in all_step_list:
        average_list.append(get_average_data(i, 100))

    labels = ["Sarsa_0", "S_L_0", "Q_learning_0"]
    images_name = "Difference for 3 0"
    all_0 = []
    for i in range(0, 6, 2):
        all_0.append(all_step_list[i])
    paint_img(episode_list, all_0, images_name, labels)

    labels = ["Sarsa_1", "S_L_1", "Q_learning_1"]
    images_name = "Difference for 3 1"
    all_1 = []
    for i in range(1, 6, 2):
        all_1.append(all_step_list[i])
    paint_img(episode_list, all_1, images_name, labels)

    labels = ["Sarsa_0", "Sarsa_1"]
    images_name = "Difference for Sarsa"
    sarsas = []
    for i in range(0, 2):
        sarsas.append(all_step_list[i])
    paint_img(x, sarsas, images_name, labels)

    labels = ["Sarsa_lambda_0", "Sarsa_lambda_1"]
    images_name = "Difference for Sarsa_lambda"
    sarsas = []
    for i in range(2, 4):
        sarsas.append(all_step_list[i])
    paint_img(x, sarsas, images_name, labels)

    labels = ["Q_learning_0", "Q_learning_1"]
    images_name = "Difference for Q learning"
    sarsas = []
    for i in range(4, 6):
        sarsas.append(all_step_list[i])
    paint_img(x, sarsas, images_name, labels)

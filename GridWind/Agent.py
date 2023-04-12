import numpy as np
from GridWind.WindGridEnvironment import *


class Agent:
    def __init__(self, env: WindGridEnv = None, capacity=10000):
        self.env = env
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None
        self.S = None
        self.A = None
        self.state = None

    def reset(self):
        self.S = []
        for i in range(self.env.grid_width):
            for j in range(self.env.grid_height):
                self.S.append((i, j))
        self.A = [i for i in range(4)]


class WindGridAgent:
    def __init__(self, env: WindGridEnv, gamma=0.9, alpha=0.1, lambda_=0.5):
        assert env is not None, "Environment is None. Please check it"
        self.state_value_function = None
        self.state_value_count = None
        self.action_value_function = None
        self.action_value_count = None
        self.state = None

        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.lambda_ = lambda_

    def reset(self, gamma=0.9, alpha=0.1, lambda_=0.5):
        self.state_value_function = np.zeros((self.env.grid_width, self.env.grid_height))
        self.state_value_count = 0
        self.action_value_function = np.zeros((self.env.grid_width, self.env.grid_height,
                                               4))
        self.action_value_count = 0
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.state = self.env.player_index

    def sarsa_algorithm(self, rounds=10000, rewards=None):
        epsilon = 1
        self.reset()

        step_list = []

        min_step = 999999
        min_episode = None

        for i in range(rounds):
            obs, _ = self.env.reset(rewards=rewards)
            state = obs
            action = epsilon_greedy_policy(self, state, epsilon)
            done = False
            episode = []

            experience_step = 0

            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = epsilon_greedy_policy(self, next_state, epsilon)

                state_x, state_y = state
                next_state_x, next_state_y = next_state

                # print("Agent action: {}".format(next_action)) print("current index: ({},{}). Next index: ({},
                # {})".format(state_x, state_y, next_state_x, next_state_y))

                current_action_value = self.action_value_function[state_x, state_y, action]
                next_action_value = self.action_value_function[next_state_x, next_state_y, next_action]
                td_target = reward + self.gamma * next_action_value
                td_error = td_target - current_action_value

                update_step = self.alpha * td_error
                self.action_value_function[state_x, state_y, action] += update_step

                state = next_state
                action = next_action

                if not done:
                    episode.append((state, action))

                experience_step += 1
            if epsilon > 0.1:
                epsilon *= 0.99
            step_list.append(experience_step)
            if experience_step < min_step:
                min_step = experience_step
                min_episode = episode

            print("round {} over. experience step: {}".format(i + 1, experience_step))
        print("The minimal step is: {}. And the min episode is: {}".format(len(min_episode), min_episode))

        return step_list

    def sarsa_lambda_algorithm(self, rounds=10000, rewards=None):
        epsilon = 0.5
        self.reset()

        step_list = []

        min_step = 999999
        min_episode = None

        E = np.zeros((self.env.grid_width, self.env.grid_height, 4))  # 引入效用迹

        for i in range(rounds):

            obs, _ = self.env.reset(rewards=rewards)
            state = obs
            action = epsilon_greedy_policy(self, state, epsilon)
            done = False
            E *= 0

            episode = []
            experience_step = 0

            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = epsilon_greedy_policy(self, next_state, epsilon)

                state_x, state_y = state
                next_state_x, next_state_y = next_state

                current_action_value = self.action_value_function[state_x, state_y, action]
                next_action_value = self.action_value_function[next_state_x, next_state_y, next_action]
                td_target = reward + self.gamma * next_action_value
                td_error = td_target - current_action_value

                E[state_x, state_y, action] += 1
                self.action_value_function += self.alpha * td_error * E
                E *= self.gamma * self.lambda_

                state = next_state
                action = next_action

                if not done:
                    episode.append((state, action))
                    experience_step += 1
            if epsilon > 0.1:
                epsilon *= 0.99
            step_list.append(experience_step)
            if experience_step < min_step:
                min_step = experience_step
                min_episode = episode
            print("round {} over. experience step: {}".format(i + 1, experience_step))
        print("The minimal step is: {}. And the min episode is: {}".format(len(min_episode), min_episode))

        return step_list


def random_policy(Agent: WindGridAgent):
    return Agent.env.action_space.sample()


def epsilon_greedy_policy(Agent: WindGridAgent, player_state, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return Agent.env.action_space.sample()
    else:
        state_x, state_y = player_state
        return np.argmax(Agent.action_value_function[state_x, state_y])

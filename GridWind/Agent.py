import numpy as np
from WindGridEnvironment import *
import time


class Agent:
    def __init__(self, env: WindGridEnv = None):
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
    def __init__(self, env: WindGridEnv, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, *, rounds=10000):
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
        self.epsilon = epsilon
        self.rounds = rounds

    def reset(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, *, rounds=10000):
        self.state_value_function = np.zeros((self.env.grid_width, self.env.grid_height))
        self.state_value_count = 0
        self.action_value_function = np.zeros((self.env.grid_width, self.env.grid_height,
                                               4))
        self.action_value_count = 0
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.state = self.env.player_index
        self.epsilon = epsilon
        self.rounds = rounds

        # print("The super parameter:")
        # print("gamma:{:.2f}, alpha:{:.2f}, lambda:{:.2f}, epsilon:{:.2f}".format(gamma, alpha, lambda_, epsilon))

    def sarsa_algorithm(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, setflag=False, *, rounds=2000, isTrain=False):
        self.reset(gamma, alpha, lambda_, epsilon, rounds=rounds)

        if setflag is False:
            rewards = (0, -1, 1)
        else:
            rewards = (-1, -1, 1)

        step_list = []

        min_step = 999999
        min_episode = None

        if isTrain is True:
            print("Training start.")

        for i in range(self.rounds):
            obs, _ = self.env.reset(rewards=rewards)
            state = obs
            action = epsilon_greedy_policy(self, state, self.epsilon)
            done = False
            episode = []
            experience_step = 0

            loop_start_time = time.time()

            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = epsilon_greedy_policy(self, next_state, self.epsilon)

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

                if time.time() - loop_start_time > 25:
                    return _, -1E15

            if self.epsilon > 0.1:
                self.epsilon *= 0.99
            step_list.append(experience_step)
            if experience_step < min_step:
                min_step = experience_step
                min_episode = episode

        #     print("round {} over. experience step: {}".format(i + 1, experience_step))
        # print("The minimal step is: {}. And the min episode is: {}".format(len(min_episode), min_episode))
        if isTrain is True:
            print("Training complete.")
            print("Playing start.")

        avg_reward = 0
        step = 0
        if isTrain is True:
            start_time = time.time()
            for i in range(10):
                obs, _ = self.env.reset(rewards=rewards)
                state = obs
                action = greedy_policy(self, state)
                done = False

                while not done:
                    next_state, reward, done, _, _ = self.env.step(action)

                    state = next_state
                    action = greedy_policy(self, state)
                    avg_reward += reward

                    step += 1
                    if isTrain is True and time.time() - loop_start_time > 25:
                        return _, -1E15

            avg_reward /= 10
            print("Playing complete.")

        return step_list, avg_reward if avg_reward > -1E15 else -1E15

    def sarsa_lambda_algorithm(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, setflag=False, *, rounds=2000, isTrain=False):
        self.reset(gamma, alpha, lambda_, epsilon)

        if setflag is False:
            rewards = (0, -1, 1)
        else:
            rewards = (-1, -1, 1)

        step_list = []

        min_step = 999999
        min_episode = None

        E = np.zeros((self.env.grid_width, self.env.grid_height, 4))  # 引入效用迹

        if isTrain is True:
            print("Training start")

        for i in range(self.rounds):

            obs, _ = self.env.reset(rewards=rewards)
            state = obs
            action = epsilon_greedy_policy(self, state, epsilon)
            done = False
            E *= 0

            episode = []
            experience_step = 0

            loop_start_time = time.time()
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

                if isTrain is True and time.time() - loop_start_time > 25:
                    return _, -1E15

            if epsilon > 0.1:
                epsilon *= 0.99
            step_list.append(experience_step)
            if experience_step < min_step:
                min_step = experience_step
                min_episode = episode
        #     print("round {} over. experience step: {}".format(i + 1, experience_step))
        # print("The minimal step is: {}. And the min episode is: {}".format(len(min_episode), min_episode))

        avg_reward = 0
        if isTrain is True:
            print("Training complete.")
            print("Playing start.")

            step = 0
            start_time = time.time()
            for i in range(10):
                obs, _ = self.env.reset(rewards=rewards)
                state = obs
                action = greedy_policy(self, state)
                done = False
                while not done:
                    next_state, reward, done, _, _ = self.env.step(action)

                    state = next_state
                    action = greedy_policy(self, state)
                    avg_reward += reward

                    if isTrain is True and time.time() - start_time > 15:
                        return _, -1E15

            avg_reward /= 10

            print("Playing complete.")

        return step_list, avg_reward if avg_reward > -1E15 else -1E15

    def Q_learning_algorithm(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.5, setflag=False, *, rounds=2000, isTrain=False):
        self.reset(gamma, alpha, lambda_, epsilon)
        if setflag is False:
            rewards = (0, -1, 1)
        else:
            rewards = (-1, -1, 1)

        step_list = []

        min_step = 999999
        min_episode = None

        if isTrain is True:
            print("Training start")

        for i in range(self.rounds):
            obs, _ = self.env.reset(rewards=rewards)
            state = obs  # 当前状态
            e_action = epsilon_greedy_policy(self, state, epsilon)  # 当前状态下采用epsilon greedy得到的行为

            done = False
            experience_step = 0
            episode = []

            loop_start_time = time.time()

            while not done:
                next_state, reward, done, _, _ = self.env.step(e_action)  # 执行行为策略的行为得到下一步状态和奖励等
                g_action = greedy_policy(self, next_state)

                state_x, state_y = state
                next_state_x, next_state_y = next_state

                current_action_value = self.action_value_function[state_x, state_y, e_action]  # 当前状态-行为对的价值
                maximum_action_value = self.action_value_function[next_state_x, next_state_y, g_action]  # 下一步的最优行为价值
                td_target = reward + self.gamma * maximum_action_value
                td_error = td_target - current_action_value
                self.action_value_function[state_x, state_y, e_action] += self.alpha * td_error

                state = next_state
                e_action = epsilon_greedy_policy(self, state, epsilon)
                if not done:
                    episode.append((state, e_action))
                    experience_step += 1

                if isTrain is True and time.time() - loop_start_time > 25:
                    return _, -1E15

            if epsilon > 0.1:
                epsilon *= 0.99

            step_list.append(experience_step)
            if experience_step < min_step:
                min_step = experience_step
                min_episode = episode
        #     print("round {} over. experience step: {}".format(i + 1, experience_step))
        # print("The minimal step is: {}. And the min episode is: {}".format(len(min_episode), min_episode))

        avg_reward = 0
        if isTrain is True:
            print("Training complete.")
            print("Playing start.")
            step = 0
            start_time = time.time()

            for i in range(10):
                obs, _ = self.env.reset(rewards=rewards)
                state = obs
                action = greedy_policy(self, state)
                done = False
                while not done:
                    next_state, reward, done, _, _ = self.env.step(action)

                    state = next_state
                    action = greedy_policy(self, state)
                    avg_reward += reward

                    if isTrain is True and time.time() - start_time > 15:
                        return _, -1E15

            avg_reward /= 10

        if isTrain is True:
            print("Playing complete.")

        return step_list, avg_reward if avg_reward > -1E15 else -1E15


def random_policy(Agent: WindGridAgent, player_state):
    return Agent.env.action_space.sample()


def greedy_policy(Agent: WindGridAgent, player_state):
    state_x, state_y = player_state
    return np.argmax(Agent.action_value_function[state_x, state_y])


def epsilon_greedy_policy(Agent: WindGridAgent, player_state, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return Agent.env.action_space.sample()
    else:
        state_x, state_y = player_state
        return np.argmax(Agent.action_value_function[state_x, state_y])

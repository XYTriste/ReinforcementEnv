import numpy as np


class BlackJackAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, *, rounds=10000):
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
        self.state_value_function = np.zeros((32, 11))
        self.state_value_count = 0
        self.action_value_function = np.zeros((32, 11,
                                               2))
        self.action_value_count = 0
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.rounds = rounds

    def sarsa_algorithm(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, setflag=False, *, rounds=10000):
        self.reset(gamma, alpha, lambda_, epsilon, rounds=rounds)

        for i in range(self.rounds):
            obs, _ = self.env.reset()
            player_state, dealer_state, _ = obs
            state = (player_state, dealer_state)
            action = epsilon_greedy_policy(self, state, self.epsilon)
            done = False
            episode = []

            while not done:
                obs, reward, done, _, _ = self.env.step(action)
                next_action = epsilon_greedy_policy(self, state, self.epsilon)

                n_p_state, n_d_state, _ = obs
                current_action_value = self.action_value_function[player_state, dealer_state, action]
                next_action_value = self.action_value_function[n_p_state, n_d_state, next_action]
                td_target = reward + self.gamma * next_action_value
                td_error = td_target - current_action_value

                self.action_value_function[player_state, dealer_state, action] += self.alpha * td_error

                player_state = n_p_state
                dealer_state = n_d_state
                action = next_action

                if not done:
                    episode.append(((player_state, dealer_state), action))
            if self.epsilon > 0.1:
                self.epsilon *= 0.99

    def sarsa_lambda_algorithm(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, setflag=False, *, rounds=2000):
        self.reset(gamma, alpha, lambda_, epsilon)

        E = np.zeros((32, 11, 2))

        for i in range(rounds):
            obs, _ = self.env.reset()
            player_state, dealer_state, _ = obs
            state = (player_state, dealer_state)
            action = epsilon_greedy_policy(self, state, self.epsilon)
            done = False
            episode = []

            E *= 0

            while not done:
                obs, reward, done, _, _ = self.env.step(action)
                next_action = epsilon_greedy_policy(self, state, self.epsilon)

                n_p_state, n_d_state, _ = obs
                current_action_value = self.action_value_function[player_state, dealer_state, action]
                next_action_value = self.action_value_function[n_p_state, n_d_state, next_action]
                td_target = reward + self.gamma * next_action_value
                td_error = td_target - current_action_value

                E[player_state, dealer_state, action] += 1
                self.action_value_function += self.alpha * td_error * E
                E *= self.gamma * self.lambda_

                player_state = n_p_state
                dealer_state = n_d_state
                action = next_action

                if not done:
                    episode.append(((player_state, dealer_state), action))
            if epsilon > 0.1:
                epsilon *= 0.99

    def Q_learning_algorithm(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.5, setflag=False, *, rounds=2000):
        self.reset(gamma, alpha, lambda_, epsilon)

        for i in range(self.rounds):
            obs, _ = self.env.reset()
            player_state, dealer_state, _ = obs
            state = (player_state, dealer_state)
            e_action = epsilon_greedy_policy(self, state, self.epsilon)

            done = False
            episode = []

            while not done:
                obs, reward, done, _, _ = self.env.step(e_action)

                n_p_state, n_d_state, _ = obs
                g_action = greedy_policy(self, (n_p_state, n_d_state))

                current_action_value = self.action_value_function[player_state, dealer_state, e_action]
                maximum_action_value = self.action_value_function[n_p_state, n_d_state, g_action]
                td_target = reward + self.gamma * maximum_action_value
                td_error = td_target - current_action_value
                self.action_value_function[player_state, dealer_state, e_action] += self.alpha * td_error

                player_state, dealer_state = n_p_state, n_d_state
                e_action = epsilon_greedy_policy(self, (player_state, dealer_state), self.epsilon)

                if not done:
                    episode.append(((player_state, dealer_state), e_action))

            if epsilon > 0.1:
                epsilon *= 0.99

    def play_with_dealer(self, rounds=10000):
        player_win = 0
        dealer_win = 0
        draw = 0
        for i in range(rounds):
            done = False

            obs, _ = self.env.reset()
            player_state, dealer_state, _ = obs
            while not done:
                action = epsilon_greedy_policy(self, (player_state, dealer_state), 0.1)
                observation, reward, done, _, _ = self.env.step(action)
                player_state = observation[0]

            if reward > 0:
                player_win += 1
            elif reward == 0:
                draw += 1
            else:
                dealer_win += 1
        return player_win, dealer_win, draw


def greedy_policy(agent: BlackJackAgent, state):
    player_state, dealer_state = state
    return np.argmax(agent.action_value_function[player_state, dealer_state])


def epsilon_greedy_policy(agent: BlackJackAgent, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return agent.env.action_space.sample()
    else:
        player_state, dealer_state = state
        return np.argmax(agent.action_value_function[player_state, dealer_state])

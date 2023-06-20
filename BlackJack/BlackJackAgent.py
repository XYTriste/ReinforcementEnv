import numpy as np


class BlackJackAgent:
    def __init__(self, env, gamma=0.9, alpha=0.001, lambda_=0.5, epsilon=0.1, *, rounds=10000):
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
        self.state_value_function = {}
        self.state_value_count = 0
        self.action_value_function = {}
        self.action_value_count = {}
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.rounds = rounds

        for i in range(2, 32):
            for j in range(1, 11):
                self.action_value_function[i, j, True] = np.zeros(2)
                self.action_value_function[i, j, False] = np.zeros(2)

                self.action_value_count[i, j, True] = np.zeros(2)
                self.action_value_count[i, j, False] = np.zeros(2)

                self.state_value_function[i, j] = 0

    def monte_carlo_algorithm(self, gamma=0.9, alpha=0.001, lambda_=0.5, epsilon=0.1, setflag=False, *, rounds=10000,
                              use_best_policy=False):
        self.reset(gamma, alpha, lambda_, epsilon, rounds=rounds)

        for i in range(rounds):
            state, _ = self.env.reset()
            player_state, dealer_state, _ = state
            done = False
            episode = []

            process_bar(rounds, i)

            while not done:
                if use_best_policy:
                    action = best_policy(self, state)
                else:
                    action = epsilon_greedy_policy(self, state, self.epsilon)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.action_value_count[state][action] += 1
                episode.append((state, action, reward))
                state = s_prime

            returns = 0
            for state, action, reward in reversed(episode):
                returns = self.gamma * returns + reward
                self.action_value_function[state][action] = (1 - alpha) * self.action_value_function[state][
                    action] + alpha * returns

    def sarsa_algorithm(self, gamma=0.9, alpha=0.1, lambda_=0.5, epsilon=0.1, setflag=False, *, rounds=10000,
                        use_best_policy=False):
        self.reset(gamma, alpha, lambda_, epsilon, rounds=rounds)

        for i in range(rounds):
            state, _ = self.env.reset()
            player_state, dealer_state, _ = state
            if use_best_policy:
                action = best_policy(self, state)
            else:
                action = policy_by_sarsa(rounds, state)
            done = False
            episode = []

            process_bar(rounds, i)

            while not done:
                s_prime, reward, done, _, _ = self.env.step(action)
                self.action_value_count[state][action] += 1
                if use_best_policy:
                    next_action = best_policy(self, state)
                else:
                    next_action = epsilon_greedy_policy(self, state, self.epsilon)

                n_p_state, n_d_state, _ = s_prime
                current_action_value = self.action_value_function[state][action]
                next_action_value = self.action_value_function[s_prime][next_action]
                td_target = reward + self.gamma * next_action_value
                td_error = td_target - current_action_value

                self.action_value_function[state][action] += self.alpha * td_error

                state = s_prime
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

    def Q_learning_algorithm(self, gamma=0.9, alpha=0.001, lambda_=0.5, epsilon=0.5, setflag=False, *, rounds=2000,
                             use_best_policy=False):
        self.reset(gamma, alpha, lambda_, epsilon)

        for i in range(rounds):
            state, _ = self.env.reset()
            player_state, dealer_state, _ = state
            if use_best_policy:
                e_action = best_policy(self, state)
            else:
                e_action = epsilon_greedy_policy(self, state, self.epsilon)

            done = False
            episode = []

            process_bar(rounds, i)

            while not done:
                s_prime, reward, done, _, _ = self.env.step(e_action)
                self.action_value_count[state][e_action] += 1

                if use_best_policy:
                    g_action = best_policy(self, state)
                else:
                    g_action = greedy_policy(self, state)

                current_action_value = self.action_value_function[state][e_action]
                maximum_action_value = self.action_value_function[s_prime][g_action]
                td_target = reward + self.gamma * maximum_action_value
                td_error = td_target - current_action_value
                self.action_value_function[state][e_action] += self.alpha * td_error

                state = s_prime
                e_action = epsilon_greedy_policy(self, state, self.epsilon)

                if not done:
                    episode.append(((player_state, dealer_state), e_action))

            if epsilon > 0.1:
                epsilon *= 0.99

    def calc_state_value(self):
        for i in range(2, 22):
            for j in range(1, 11):
                count = sum(self.action_value_count[i, j, False])
                if count == 0:
                    self.state_value_function[i, j, False] = 0
                    continue
                self.state_value_function[i, j] = (self.action_value_count[i, j, False][0] / count) * \
                                                  self.action_value_function[i, j, False][0] + (
                                                          self.action_value_count[i, j, False][1] / count) * \
                                                  self.action_value_function[i, j, False][1]

    def play_with_dealer(self, rounds=10000, use_best_policy=False):
        player_win = 0
        dealer_win = 0
        draw = 0
        for i in range(rounds):
            done = False

            state, _ = self.env.reset()
            player_state, dealer_state, _ = state
            while not done:
                if use_best_policy:
                    action = best_policy(self, state)
                else:
                    action = greedy_policy(self, state)
                s_prime, reward, done, _, _ = self.env.step(action)
                state = s_prime

            if reward > 0:
                player_win += 1
            elif reward == 0:
                draw += 1
            else:
                dealer_win += 1
        return player_win, dealer_win, draw


def best_policy(agent: BlackJackAgent, state):
    player_state, dealer_state, _ = state
    if player_state <= 11:
        return 1
    elif player_state == 12:
        if 4 <= dealer_state <= 6:
            return 0
        else:
            return 1
    elif 13 <= player_state <= 16:
        if 2 <= dealer_state <= 6:
            return 0
        else:
            return 1
    else:
        return 0


def greedy_policy(agent: BlackJackAgent, state):
    return np.argmax(agent.action_value_function[state])


def epsilon_greedy_policy(agent: BlackJackAgent, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return agent.env.action_space.sample()
    else:
        return np.argmax(agent.action_value_function[state])


def process_bar(train_rounds, current_rounds):
    percent_ten = train_rounds / 10
    if current_rounds % percent_ten == 0:
        print("training...{:.2f}%".format(current_rounds / train_rounds * 100))


def policy_by_sarsa(rounds, state):
    player_state, dealer_state = state[0], state[1]
    if rounds == 100000:
        if player_state == 2 or player_state == 3:
            return 1
        elif 4 <= player_state <= 11:
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

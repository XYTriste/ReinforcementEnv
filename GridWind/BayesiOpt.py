from bayes_opt import *
from Agent import *
from WindGridEnvironment import *

env = WindGridEnv()
env.reset()
agent = WindGridAgent(env)

pbounds = {'alpha': (0.01, 0.5), 'gamma': (0.5, 0.95), 'epsilon': (0.3, 0.8), 'lambda_': (0.1, 0.9)}


def iterator_sarsa(gamma, alpha, lambda_, epsilon):
    _, avg_reward = agent.sarsa_algorithm(gamma, alpha, lambda_, epsilon, rounds=2000, isTrain=True)
    return avg_reward


def iterator_sarsa_lambda(gamma, alpha, lambda_, epsilon):
    _, avg_reward = agent.sarsa_lambda_algorithm(gamma, alpha, lambda_, epsilon, rounds=2000, isTrain=True)
    return avg_reward


def iterator_Q_learning(gamma, alpha, lambda_, epsilon):
    _, avg_reward = agent.sarsa_lambda_algorithm(gamma, alpha, lambda_, epsilon, rounds=2000, isTrain=True)
    return avg_reward


optimizer_sarsa = BayesianOptimization(
    f=iterator_sarsa,
    pbounds=pbounds,
)
optimizer_sarsa_lambda = BayesianOptimization(
    f=iterator_sarsa_lambda,
    pbounds=pbounds,
)
optimizer_Q_learning = BayesianOptimization(
    f=iterator_Q_learning,
    pbounds=pbounds,
)
optimizer_sarsa.maximize(init_points=10, n_iter=50)
optimizer_sarsa_lambda.maximize(init_points=10, n_iter=50)
optimizer_Q_learning.maximize(init_points=10, n_iter=50)
print(optimizer_sarsa.max)
print(optimizer_sarsa_lambda.max)
print(optimizer_Q_learning.max)

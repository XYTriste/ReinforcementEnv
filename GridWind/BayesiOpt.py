from bayes_opt import *
from Agent import *
from WindGridEnvironment import *

env = WindGridEnv()
env.reset()
agent = WindGridAgent(env)

pbounds = {'alpha': (0.1, 0.2), 'gamma': (0.8, 0.95), 'epsilon': (0.3, 0.5), 'lambda_': (0.3, 0.9)}


def iteratorFunction(gamma, alpha, lambda_, epsilon):
    _, avg_reward = agent.sarsa_algorithm(gamma, alpha, lambda_, epsilon, rounds=2000, isTrain=True)
    return avg_reward if avg_reward > -np.inf else -np.inf


optimizer = BayesianOptimization(
    f=iteratorFunction,
    pbounds=pbounds,
)
optimizer
optimizer.maximize(init_points=10, n_iter=30)
print(optimizer.max)

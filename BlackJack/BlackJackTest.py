from BlackJackEnv_v1 import *
from BlackJackAgent import *

env = BlackJackEnvironment()
agent = BlackJackAgent(env)
rounds = 10000

agent.Q_learning_algorithm(rounds=rounds, epsilon=0.9)
print("Training complete")
info = agent.play_with_dealer()
print("Player win rate: {:.2f}%   Dealer win rate:{:.2f}%   Not lose rate:{:.2f}%"
      .format(info[0] / rounds * 100, info[1] / rounds * 100, (rounds - info[1]) / rounds * 100))

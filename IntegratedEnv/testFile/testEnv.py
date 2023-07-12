import gymnasium
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    env = gymnasium.make("MontezumaRevengeNoFrameskip-v4", render_mode="human")

    state, info = env.reset()
    print(env.unwrapped.ale.lives())
    # while True:
    #     action = env.action_space.sample()
    #     print("action", action)
    #     plt.imshow(state)
    #     plt.axis('off')
    #     plt.show()
    #     s_prime, reward, done, _, _ = env.step(action)
    #     # time.sleep(3)
    #     if done:
    #         state, info = env.reset()
    #         state = state[20:, :]
    #         print(info)
    #     else:
    #         state = s_prime
    #         state = state[20:, :]


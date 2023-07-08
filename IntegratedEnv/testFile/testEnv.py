import gymnasium
import time

if __name__ == '__main__':
    env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode="human")

    state, info = env.reset()
    while True:
        action = env.action_space.sample()
        print("action", action)
        s_prime, reward, done, _, _ = env.step(action)
        time.sleep(3)
        if done:
            state, info = env.reset()
        else:
            state = s_prime


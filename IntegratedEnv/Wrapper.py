import multiprocessing
import multiprocessing.connection

import cv2
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from Tools import SetupArgs, Painter


class Game:
    def __init__(self, args: SetupArgs, seed, gameNumber):
        self.gameNumber = gameNumber
        self.env_name = args.env_name
        self.env = gymnasium.make(args.env_name)
        self.env.seed(seed)
        self.obs_4 = np.zeros((4, 84, 84))
        self.obs_2_max = np.zeros((2, 84, 84))
        self.returns = []
        self.rewards = []
        self.lives = 5  # Atari游戏中的生命值参数，该参数需要根据环境进行调整,但是reset方法中会动态调整。

        self.width_start = args.obs_cut['width_start']
        self.width_end = args.obs_cut['width_end']
        self.height_start = args.obs_cut['height_start']
        self.height_end = args.obs_cut['height_end']
        self.reward_cut = args.reward_cut

    def step(self, action):
        reward = 0.
        done = False
        for i in range(4):
            s_prime, r, done, info, _ = self.env.step(action)
            s_prime = s_prime[self.width_start: self.width_end, self.height_start: self.height_end]
            # plt.imshow(s_prime)
            # plt.axis('off')
            # plt.show()

            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(s_prime)
            reward += r
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives:
                done = True
                break

        reward *= self.reward_cut
        self.rewards.append(reward)

        if done:
            self.returns.append(sum(self.rewards))
            episode_info = {
                "reward": sum(self.rewards),
                "length": len(self.rewards)
            }
            self.reset()
        else:
            episode_info = None
            obs = self.obs_2_max.max(axis=0)
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward, done, episode_info, {}

    def reset(self):
        obs, info = self.env.reset()
        obs = obs[self.width_start: self.width_end, self.height_start: self.height_end]
        obs = self._process_obs(obs)
        for i in range(4):
            self.obs_4[i] = obs
        self.rewards = []
        self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4, info

    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def worker_process(remote: multiprocessing.connection.Connection, args: SetupArgs, seed, gameNumber):
    game = Game(args, seed, gameNumber)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    def __init__(self, args: SetupArgs, seed, gameNumber):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, args, seed, gameNumber))
        self.process.start()


if __name__ == '__main__':
    env = gymnasium.make("MountainCar-v0")
    print(env.unwrapped.lives())

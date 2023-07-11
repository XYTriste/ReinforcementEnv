import gymnasium
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import time


class SetupArgs:
    def __init__(self):
        pass

    def get_args(self, description="Parameters setting"):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
        parser.add_argument('--num_episodes', type=int, default=1500, help='Training frequency')
        parser.add_argument('--seed', type=int, default=24, metavar='S', help='set random seed')
        parser.add_argument("--gamma", type=float, default=0.95, metavar='S', help='discounted rate')
        parser.add_argument('--epsilon', type=float, default=1, metavar='S', help='Exploration rate')
        parser.add_argument('--buffer_size', type=int, default=2 ** 16, metavar='S',
                            help='Experience replay buffer size')
        parser.add_argument('--env_name', type=str, default="ALE/MontezumaRevenge-v5", metavar='S', help="Environment name")

        return parser.parse_args()


class Game:
    def __init__(self, args: SetupArgs, seed):
        self.env_name = args.env_name
        self.env = gymnasium.make(args.env_name, render_mode=args.render_mode)
        self.env.seed(seed)
        self.obs_4 = np.zeros((4, 84, 84))
        self.obs_2_max = np.zeros((2, 84, 84))
        self.returns = []
        self.rewards = []
        self.frames = 0
        self.lives = 5  # Atari游戏中的生命值参数，该参数需要根据环境进行调整,但是reset方法中会动态调整。

        self.width_start = args.obs_cut['width_start']
        self.width_end = args.obs_cut['width_end']
        self.height_start = args.obs_cut['height_start']
        self.height_end = args.obs_cut['height_end']
        self.reward_cut = args.reward_cut

    def step(self, action):
        reward = 0.
        done = False
        # print('action:', action)
        for i in range(1):
            s_prime, r, done, info, _ = self.env.step(action)
            s_prime = s_prime[self.width_start: self.width_end, self.height_start: self.height_end]
            # plt.imshow(s_prime)
            # plt.axis('off')
            # plt.show()

            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(s_prime)
            reward += r
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives or self.frames > 100000:
                # print('frames:', self.frames)
                done = True
                break
        self.obs_2_max[0] = self._process_obs(s_prime)
        reward *= self.reward_cut
        self.rewards.append(reward)
        self.frames += 1

        if done:
            self.returns.append(sum(self.rewards))
            episode_info = {
                "reward": sum(self.rewards),
                "length": len(self.rewards),
                'frames': self.frames
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
        self.frames = 0

        return self.obs_4, info

    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


if __name__ == '__main__':
    args = SetupArgs().get_args()

    args.INPUT_DIM = 4
    args.HIDDEN_DIM = 128
    args.OUTPUT_DIM = 5
    args.HIDDEN_DIM_NUM = 5
    args.obs_cut = {
        'width_start': 0,
        'width_end': 210,
        'height_start': 0,
        'height_end': 160
    }
    args.render_mode="human"
    args.reward_cut = 1

    game = Game(args, 21)
    state, info = game.reset()
    for i in range(3):
        action = 4
        s_prime, reward, done, info, _ = game.step(action)
        state = s_prime
        # time.sleep(2)
    # for i in range(4):
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(state[i])
    #     # plt.colorbar()
    #     plt.show()
    motion_tensor = np.zeros((3, 84, 84))
    for i in range(1, 4):
        frame_diff = np.abs(state[i] - state[i - 1])
        threshold = 10
        motion_tensor[i - 1] = np.where(frame_diff > threshold, 1, 0)
    for i in range(1, 3):
        print(np.mean((motion_tensor[i - 1] - motion_tensor[i]) ** 2))
    for i in range(3):
        plt.figure(figsize=(6, 6))
        plt.imshow(motion_tensor[i], cmap="hot")
        plt.colorbar()
        plt.show()

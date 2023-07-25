import gymnasium
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from scipy.stats import entropy


def mean_squared_error(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse


def kl_divergence(image1, image2):
    # 将图像像素值规范化为概率分布
    prob1 = (image1.flatten() + 1e-10) / np.sum(image1)
    prob2 = (image2.flatten() + 1e-10) / np.sum(image2)

    # 使用SciPy库的entropy函数计算KL散度
    kl_div = entropy(prob1, qk=prob2)
    return kl_div


def _process_obs(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs


if __name__ == '__main__':
    env = gymnasium.make("MontezumaRevengeNoFrameskip-v4", render_mode="human")
    maxmean = 0
    minmean = 999
    maxkl = 0
    minkl = 999

    i = 0

    state, info = env.reset()
    while i < 2000:
        action = 3
        s_prime, reward, done, _, _ = env.step(action)

        c_s = _process_obs(state)
        n_s = _process_obs(s_prime)

        mean = mean_squared_error(c_s, n_s)
        kl = kl_divergence(c_s, n_s)
        maxmean = max(mean, maxmean)
        maxkl = max(maxkl, kl)
        minmean = min(minmean, mean)
        minkl = min(minkl, kl)


        print("mean:", mean, " kl:", kl)


        if done:
            state, info = env.reset()

        state = s_prime

        i += 1

    print("Max mean:{}   Min mean:{}    Max KL:{}   Min KL:{}".format(maxmean, minmean, maxkl, minkl))
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

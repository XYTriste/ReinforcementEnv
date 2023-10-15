import numpy as np
import torch.cuda

from MazeEnv import *
from DQN import *
from RND import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
def random_policy():
    return random.randint(0, 3)


class Recorder:
    def __init__(self):
        self.episodes = []
        self.time_steps = []
        self.episode_rewards = []
        self.predict_errors = []
        self.DQNLoss = []


if __name__ == '__main__':
    env = MazeEnv(30, 30, render_mode="rgb_array")
    agent = DQN(env)
    rnd = RND(2, 1)
    rounds = 1000

    reward_weight = 0.01

    recorder = Recorder()

    writer = SummaryWriter(log_dir='./log')

    agent_learn_step = 0  # 记录总步数，每隔一段时间进行经验回放

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)

    for i in range(10):
        with tqdm(total=int(rounds / 10), desc=f'Iteration {i}') as pbar:
            for episode in range(rounds // 10):
                state, _ = env.reset()

                episode_extrinsic_reward = 0
                episode_intrinsic_reward = 0
                episode_reward = 0
                episode_predict_error = 0
                episode_DQN_loss = 0
                episode_learn_count = 0  # 每回合DQN算法学习的次数

                time_step = 0
                done = False
                while not done:
                    action = agent.epsilon_greedy_policy(state)

                    s_prime, extrinsic_reward, done, _, _ = env.step(action)
                    s_prime_tensor = torch.tensor(torch.from_numpy(s_prime), device=device, dtype=torch.float)
                    predict, target = rnd(s_prime_tensor)
                    intrinsic_reward = rnd.update_parameters(predict, target)

                    reward = (1 - reward_weight) * extrinsic_reward + reward_weight * intrinsic_reward

                    episode_extrinsic_reward += extrinsic_reward
                    episode_intrinsic_reward += reward_weight * intrinsic_reward
                    episode_reward += reward

                    agent.store_transition(state, action, reward, s_prime, done)

                    state = s_prime

                    time_step += 1
                    if agent.memory_counter > agent.MEMORY_CAPACITY and agent_learn_step % 5 == 0:
                        episode_DQN_loss += agent.learn()

                writer.add_scalar("Episode intrinsic reward", episode_intrinsic_reward, i * (rounds // 10) + episode)
                writer.add_scalar("Episode extrinsic reward", episode_extrinsic_reward, i * (rounds // 10) + episode)
                writer.add_scalar("Episode reward", episode_reward, i * (rounds // 10) + episode)
                writer.add_scalar("Episode DQN Loss", episode_DQN_loss, i * (rounds // 10) + episode)
                writer.add_scalar("Time step", time_step, i * (rounds // 10) + episode)

                if agent.EPSILON > 0.01:
                    agent.EPSILON *= 0.99

                recorder.episodes.append(i)
                recorder.time_steps.append(time_step)
                recorder.episode_rewards.append(reward)
                recorder.predict_errors.append(episode_predict_error)
                recorder.DQNLoss.append(episode_DQN_loss / (episode_learn_count if episode_learn_count != 0 else 1))

                if (episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": f"{rounds / 10 * i + episode + 1}",
                            "return": f"{np.mean(recorder.episode_rewards[-10:]):3f}"
                        }
                    )
                pbar.update(1)

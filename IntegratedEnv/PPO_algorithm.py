from typing import Dict
from Tools import *
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from Network import *

from labml import monit, tracker, logger, experiment
from labml.configs import FloatDynamicHyperParam, IntDynamicHyperParam
from Wrapper import *
from labml_nn.rl.ppo import ClippedPPOLoss, ClippedValueFunctionLoss
from labml_nn.rl.ppo.gae import GAE

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=INPUT_DIM, out_channels=32, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.pi_logits = nn.Linear(in_features=512, out_features=OUTPUT_DIM)

        self.value = nn.Linear(in_features=512, out_features=1)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.reshape((-1, 7 * 7 * 64))

        x = self.activation(self.lin(x))

        pi = Categorical(logits=self.pi_logits(x))
        value = self.value(x).reshape(-1)

        return pi, value


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class PPOTrainer:
    def __init__(self, *,
                 updates: int,
                 epochs: IntDynamicHyperParam,
                 n_workers: int,
                 worker_steps: int,
                 batches: int,
                 value_loss_coef: FloatDynamicHyperParam,
                 entropy_bonus_coef: FloatDynamicHyperParam,
                 clip_range: FloatDynamicHyperParam,
                 learning_rate: FloatDynamicHyperParam,
                 args: SetupArgs,
                 test: dict):
        self.args = args
        self.INPUT_DIM = args.INPUT_DIM  # 输入层的大小
        self.HIDDEN_DIM = args.HIDDEN_DIM  # 隐藏层的大小
        self.OUTPUT_DIM = args.OUTPUT_DIM  # 输出层的大小
        self.HIDDEN_DIM_NUM = args.HIDDEN_DIM_NUM  # 隐藏层的数量

        self.algorithm_name = "PPO"

        self.test = test['use_test']    # 是否加载模型并进行测试
        self.test_model = test['test_model']


        """----------RND网络参数定义部分----------"""
        self.use_rnd = args.rnd['use_rnd']
        self.rnd_weight = args.rnd['rnd_weight']
        self.rnd_weight_decay = args.rnd['rnd_weight_decay']
        """----------RND网络参数定义结束----------"""

        """--------------------------------RND网络的定义部分--------------------------------"""
        if self.use_rnd:
            self.RND_Network = RNDNetwork_CNN(args).to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.RND_Network = None
        """--------------------------------RND网络的定义结束--------------------------------"""

        self.updates = updates

        self.epochs = epochs

        self.n_workers = n_workers

        self.worker_steps = worker_steps

        self.batches = batches

        self.batch_size = self.n_workers * self.worker_steps

        self.mini_batch_size = self.batch_size // self.batches
        assert (self.batch_size % self.batches == 0)

        self.value_loss_coef = value_loss_coef

        self.entropy_bonus_coef = entropy_bonus_coef

        self.clip_range = clip_range

        self.learning_rate = learning_rate

        self.workers = [Worker(args, 47 + i, i) for i in range(self.n_workers)]
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)

        for worker in self.workers:
            worker.child.send(("reset", None))

        for i, worker in enumerate(self.workers):
            recv, info = worker.child.recv()
            self.obs[i] = recv

        self.model = Model(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4)

        self.gae = GAE(self.n_workers, self.worker_steps, 0.99, 0.95)  # 广义优势估计函数，用于计算PPO损失函数中对动作值的优势估计

        self.ppo_loss = ClippedPPOLoss()

        self.value_loss = ClippedValueFunctionLoss()

        self.painter = Painter()
        self.returns = []
        self.all_returns = []  # 把所有线程中得到的结果进行保存
        self.watch_processing = 3  # 指定绘制第几个线程的输出结果
        for i in range(self.n_workers):
            self.returns.append([])
        self.all_frames = 0     # 记录总共训练帧数

    def sample(self) -> Dict[str, torch.Tensor]:
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)

        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)

        done = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool_)

        obs = np.zeros((self.n_workers, self.worker_steps, 4, 84, 84), dtype=np.uint8)

        log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)

        values = np.zeros((self.n_workers, self.worker_steps + 1), dtype=np.float32)

        for t in range(self.worker_steps):
            obs[:, t] = self.obs

            pi, v = self.model(obs_to_torch(self.obs))
            values[:, t] = v.detach().cpu().numpy()
            a = pi.sample()
            actions[:, t] = a.cpu().numpy()
            log_pis[:, t] = pi.log_prob(a).detach().cpu().numpy()

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):
                self.obs[w], rewards[w, t], done[w, t], info, _ = worker.child.recv()
                # self.returns[w].append(rewards[w, t])

                if self.use_rnd:
                    predict, target = self.RND_Network(self.obs[w])
                    intrinsic_reward = self.RND_Network.get_intrinsic_reward(predict, target)
                    rewards[w, t] += self.rnd_weight * intrinsic_reward

                if info:
                    self.all_returns.append(info['reward'])
                    self.all_frames += info['frames']
                    tracker.add('reward', info['reward'])
                    tracker.add('length', info['length'])

        _, v = self.model(obs_to_torch(self.obs))
        values[:, self.worker_steps] = v.detach().cpu().numpy()

        advantages = self.gae(done, rewards, values)

        samples = {
            'obs': obs,
            'actions': actions,
            'values': values[:, :-1],
            'log_pis': log_pis,
            'advantages': advantages
        }

        samples_flat = {}

        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def train(self, samples: Dict[str, torch.Tensor]):
        for _ in range(self.epochs()):
            indexes = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}

                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                loss = self._calc_loss(mini_batch)

                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate()

                self.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                self.optimizer.step()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        sampled_return = samples['values'] + samples['advantages']

        sampled_normalized_advantage = self._normalize(samples['advantages'])

        pi, value = self.model(samples['obs'])

        log_pi = pi.log_prob(samples['actions'])

        policy_loss = self.ppo_loss(log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range())

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        value_loss = self.value_loss(value, samples['values'], sampled_return, self.clip_range())

        loss = (policy_loss + self.value_loss_coef() * value_loss - self.entropy_bonus_coef() * entropy_bonus)

        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()

        tracker.add({
            'policy_reward': policy_loss,
            'value_loss': value_loss,
            'entropy_bonus': entropy_bonus,
            'kl_div': approx_kl_divergence,
            'clip_fraction': self.ppo_loss.clip_fraction
        })

        return loss

    def run_training_loop(self):
        tracker.set_queue('reward', 100, True)
        tracker.set_queue('length', 100, True)

        for update in monit.loop(self.updates):
            samples = self.sample()

            self.train(samples)

            tracker.save()

            if (update + 1) % 1000 == 0:
                logger.log()
                self.save_info(message="{}r-{}-{}".format(update + 1, self.all_frames, "_RND" if self.use_rnd else ""))

        self.save_info(message="final-{}-{}".format(self.all_frames, "_RND" if self.use_rnd else ""))

    def save_info(self, message=""):
        formatted_time = datetime.now().strftime("%y_%m_%d_%H")
        env_name = self.args.env_name.split("/")[-1]
        torch.save(self.model.state_dict(), "./checkpoint/{}_{}_{}_{}".format(self.algorithm_name, env_name, formatted_time, message))

        # for i in range(self.n_workers):
        #     fileName = './data/{}_{}_Process_{}_{}_{}.txt'.format(self.algorithm_name, env_name, i, formatted_time, message)
        #     with open(fileName, 'w') as file_object:
        #         file_object.write(str(self.returns[i]))

        fileName = './data/{}_{}_All Process_{}_{}.txt'.format(self.algorithm_name, env_name, message, formatted_time)
        with open(fileName, 'w') as file_object:
            file_object.write(str(self.all_returns))


    def destroy(self):
        for worker in self.workers:
            worker.child.send(('close', None))

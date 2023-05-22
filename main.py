import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import count
import gym
import time

import matplotlib
matplotlib.use("TkAgg")


class EGreedyExpStrategy():  # epsilons-greedy strategy
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=1000000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, q_values, state):

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
            self.exploratory_action_taken = False
        else:
            action = np.random.randint(len(q_values))
            self.exploratory_action_taken = True

        self._epsilon_update()
        return action


def get_discrete_state(state):  # get the state from environment and transfer the observation into integer
    DISCRETE_OS_SIZE = [20, 20]
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))


def plot(train_values, eval_values, moving_avg_period):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(train_values, color="black", label="training")

    training_avg = get_moving_average(moving_avg_period, train_values)
    evaluation_avg = get_moving_average(moving_avg_period, eval_values)
    plt.plot(training_avg, color="blue", label="train avg")
    plt.plot(evaluation_avg, color="red", label="eval avg")
    plt.legend(loc="upper left")
    plt.pause(0.001)
    print(f"Episode {len(train_values)} \n"
          f" {moving_avg_period} episode train avg: {training_avg[-1]} \n"
          f" {moving_avg_period} episode evaluation avg: {evaluation_avg[-1]} \n")


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values)) + 200
        return moving_avg.numpy()


class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)

class DQN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(DQN, self).__init__()
        # 初始化及一些超参数
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # 确保输入为tensor
    def _format(self, state):
        x = state[0]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x),
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    # 前馈
    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

    # 数据记录
    def load(self, experiences):
        states, actions, rewards, new_states, is_dones = experiences
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        new_states = torch.tensor(new_states).float().to(self.device)
        is_dones = torch.tensor(is_dones).float().to(self.device)
        return states, actions, rewards, new_states, is_dones

class ReplayBuffer():
    def __init__(self,
                 max_size=10000,
                 batch_size=64):
        self.ss_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.as_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.rs_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.ps_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.ds_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        # state, action, reward, next state, done
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        self._idx += 1
        self._idx = self._idx % self.max_size
        self.size += 1
        self.size = min(self.size, self.max_size)

    # 随机采样
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
                      np.vstack(self.as_mem[idxs]), \
                      np.vstack(self.rs_mem[idxs]), \
                      np.vstack(self.ps_mem[idxs]), \
                      np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        return self.size

class AgentDQN:
    def __init__(self,
                 env,
                 training_strategy,
                 evaluation_strategy,
                 replay_buffer,
                 online_model,
                 target_model,
                 optimizer,
                 lr,
                 gamma,
                 n_warmup_batches,
                 update_target_every_steps,
                 max_episodes,
                 moving_average_period
                 ):
        self.env = env
        self.training_strategy = training_strategy
        self.evaluation_strategy = evaluation_strategy
        self.replay_buffer = replay_buffer
        self.online_model = online_model
        self.target_model = target_model
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(online.data)
        self.optimizer = optimizer
        self.value_optimizer = optimizer(self.online_model.parameters(), lr=lr)
        self.gamma = gamma
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.max_episodes = max_episodes
        self.moving_average_period = moving_average_period
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []
        self._loss = 0.0

    # 执行一次step，返回新的state和游戏是否结束
    # 这里使用了一个trick，修改了gym中反馈的原始奖励
    def interaction_step(self, state, env):
        q_values = self.online_model(state).detach().cpu().data.numpy().squeeze()
        action = self.training_strategy.select_action(q_values, state)
        new_state, reward, is_done, _, _ = env.step(action)
        reward = 20 if new_state[0] >= 0.5 else abs(new_state[0] - state[0])
        is_failure = is_done
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_done

    # 反向优化
    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_dones = experiences

        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_dones))
        q_sa = self.online_model(states).gather(1, actions)

        # td_error = q_sa - target_q_sa
        # value_loss = td_error.pow(2).mul(0.5).mean()
        value_loss = F.mse_loss(q_sa, target_q_sa)
        self._loss = value_loss
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    # 张建过程过节的评估，只是为了看看效果如何
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        step = []
        for _ in range(n_episodes):
            s = eval_env.reset()
            d = False
            rs.append(0)
            step.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                step[-1] += 1
                if d : break
        return np.mean(step), np.std(step)


# 构造DQN算法类
agent = AgentDQN(
    env=gym.make("MountainCar-v0"),
    training_strategy=EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.01, decay_steps=50000),
    evaluation_strategy=GreedyStrategy(),
    replay_buffer=ReplayBuffer(max_size=100000, batch_size=64),
    online_model=DQN(2, 3, hidden_dims=(128, 128)),
    target_model=DQN(2, 3, hidden_dims=(128, 128)),
    optimizer=optim.RMSprop,
    lr=0.001,
    gamma=0.99,
    n_warmup_batches=5,
    update_target_every_steps=30,
    max_episodes=500,
    moving_average_period=50
)

# 正式训练，训练epicode轮
for episode in range(1, agent.max_episodes + 1):
    # 初始状态
    agent.env.reset()
    num_states = [20, 20]
    is_done = False
    agent.env.render()
    agent.episode_reward.append(0.0)
    agent.episode_timestep.append(0.0)
    agent.episode_exploration.append(0.0)
    # print( agent.replay_buffer.batch_size * agent.n_warmup_batches)
    state = agent.env.reset()
    for step in count():
        # 执行一次动作
        state, is_done = agent.interaction_step(state, agent.env)
        min_samples = agent.replay_buffer.batch_size * agent.n_warmup_batches
        # ubffer中的长度大于最小采样数，可以更新
        if len(agent.replay_buffer) > min_samples:
            # 采样，加载数据，反传优化模型
            experiences = agent.replay_buffer.sample()
            experiences = agent.online_model.load(experiences)
            agent.optimize_model(experiences)

        # 参数替换
        if np.sum(agent.episode_timestep) % agent.update_target_every_steps == 0:
            for target, online in zip(agent.target_model.parameters(), agent.online_model.parameters()):
                target.data.copy_(online.data)
        # 判断一次是否结束
        if is_done:
            evaluation_score, _ = agent.evaluate(agent.online_model, agent.env, n_episodes=5)
            agent.evaluation_scores.append(evaluation_score)
            plot(agent.episode_timestep, agent.evaluation_scores, agent.moving_average_period)
            break

for episode in range(5):
    env = gym.make("MountainCar-v0", render_mode="human")  # "LunarLander-v2"
    s, info = env.reset()
    d = False
    duration = 0.0
    time.sleep(1)

    for step in count():
        env.render()
        time.sleep(0.05)

        a = agent.evaluation_strategy.select_action(agent.online_model, s)
        s, r, d, trun, _ = env.step(a)
        duration += 1

        if d or trun:
            print(f"Episode {episode + 1}: took {duration} steps")
            time.sleep(1)
            env.close()
            break
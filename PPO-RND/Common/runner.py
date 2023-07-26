from Common.utils import *
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class ANet(nn.Module):
    def __init__(self, state_shape):
        super(ANet, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.encoded_features = nn.Linear(in_features=flatten_size, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.encoded_features(x)


class Worker:
    def __init__(self, id, **config):
        self.id = id
        self.config = config
        self.env_name = self.config["env_name"]
        self.max_episode_steps = self.config["max_frames_per_episode"]
        self.state_shape = self.config["state_shape"]
        self.env = make_atari(self.env_name, self.max_episode_steps)
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self._stacked_states_info = np.zeros(self.state_shape, dtype=np.uint8)
        self.reset()

        # self.initial = ANet()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state, _ = self.env.reset()
        self._stacked_states, self._stacked_states_info, _ = stack_states(self._stacked_states, self._stacked_states_info, state, True)

    def step(self, conn):
        t = 1
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, r, d, _, info = self.env.step(action)
            t += 1
            if t % self.max_episode_steps == 0:
                d = True
            if self.config["render"]:
                self.render()
            self._stacked_states, self._stacked_states_info, efficient = stack_states(self._stacked_states, self._stacked_states_info, next_state, False)
            conn.send((self._stacked_states, np.sign(r), d, info, efficient))
            if d:
                self.reset()
                t = 1

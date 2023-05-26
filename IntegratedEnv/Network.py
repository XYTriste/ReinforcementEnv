# -*- coding = utf-8 -*-
# @time: 5/26/2023 9:05 PM
# Author: Yu Xia
# @File: Network.py
# @software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, HIDDEN_LAYERS_NUM=1):
        super(Net, self).__init__()
        self.HIDDEN_LAYERS_NUM = HIDDEN_LAYERS_NUM

        self.input_layer = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.hidden_layers = nn.ModuleList()
        for i in range(HIDDEN_LAYERS_NUM):
            layer = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.hidden_layers.append(layer)
        self.output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x
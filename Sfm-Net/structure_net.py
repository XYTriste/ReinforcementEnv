import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_deconv_net import ConvDeconvNet  # Ensure this is adapted to PyTorch


class StructureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cd_net = ConvDeconvNet()
        self.depth = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)  # Assuming 512 input channels

    def forward(self, x):
        x, _ = self.cd_net(x)
        depth = torch.sigmoid(self.depth(x)) * 99 + 1
        pc = depth_to_point(depth)
        return depth, pc


def depth_to_point(depth, camera_intrinsics=(0.5, 0.5, 1.0)):
    cx, cy, cf = camera_intrinsics
    b, c, h, w = depth.size()

    x_l = torch.linspace(-cx, 1 - cx, w, device=depth.device) / cf
    y_l = torch.linspace(-cy, 1 - cy, h, device=depth.device) / cf

    y, x = torch.meshgrid(y_l, x_l)
    f = torch.ones_like(x)

    grid = torch.stack([x, y, f], -1).unsqueeze(0).repeat(b, 1, 1, 1)
    return depth * grid

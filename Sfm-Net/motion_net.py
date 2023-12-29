import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_deconv_net import ConvDeconvNet


class MotionNet(nn.Module):
    def __init__(self, num_masks=3):
        super(MotionNet, self).__init__()
        self.num_masks = num_masks

        self.cd_net = ConvDeconvNet()
        self.obj_mask = nn.Conv2d(32, num_masks, 1)

        self.d1 = nn.Linear(1024, 512)
        self.d2 = nn.Linear(512, 512)

        self.cam_t = nn.Linear(512, 3)
        self.cam_p = nn.Linear(512, 600)
        self.cam_r = nn.Linear(512, 3)

        self.obj_t = nn.Linear(512, 3 * num_masks)
        self.obj_p = nn.Linear(512, 600 * num_masks)
        self.obj_r = nn.Linear(512, 3 * num_masks)

    def forward(self, f0, f1, sharpness_multiplier):
        x = torch.cat([f0, f1], dim=1)
        x, r = self.cd_net(x)
        b, *_ = r.shape

        r = r.view(b, -1)
        r = F.relu(self.d1(r))
        r = F.relu(self.d2(r))

        obj_mask = torch.sigmoid(self.obj_mask(x) * sharpness_multiplier)

        obj_t = self.obj_t(r)
        obj_p = self.obj_p(r)
        obj_p = obj_p.view(-1, self.num_masks, 600)
        obj_p = F.softmax(obj_p, dim=2)
        obj_r = torch.tanh(self.obj_r(r))

        cam_t = self.cam_t(r)
        cam_p = F.softmax(self.cam_p(r), dim=1)
        cam_r = torch.tanh(self.cam_r(r))

        return (obj_mask, obj_t, obj_p, obj_r), (cam_t, cam_p, cam_r)

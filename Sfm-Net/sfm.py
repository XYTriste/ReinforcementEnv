import torch
import torch.nn as nn
import numpy as np
import math

from structure_net import StructureNet
from motion_net import MotionNet


class SfMNet(nn.Module):
    def __init__(self):
        super(SfMNet, self).__init__()

        self.structure = StructureNet()
        self.motion = MotionNet()

    def forward(self, f0, f1, sharpness_multiplier):
        depth, pc = self.structure(f0)
        obj_params, cam_params = self.motion(f0, f1, sharpness_multiplier)
        motion_maps, pc_t = apply_obj_transform(pc, *obj_params)
        pc_t = apply_cam_transform(pc_t, *cam_params)
        points, flow = optical_flow(pc_t)
        return depth, points, flow, obj_params, cam_params, pc_t, motion_maps


def apply_obj_transform(pc, obj_mask, obj_t, obj_p, obj_r, num_masks=3):
    b, h, w, c = pc.shape

    p = _pivot_point(obj_p)
    R = _r_mat(obj_r.view(-1, 3))

    p = p.view(b, 1, 1, num_masks, 3)
    t = obj_t.view(b, 1, 1, num_masks, 3)
    R = R.view(b, 1, 1, num_masks, 3, 3).expand(-1, h, w, -1, -1, -1)

    pc = pc.view(b, h, w, 1, 3)
    mask = obj_mask.view(b, h, w, num_masks, 1)

    pc_t = pc - p
    pc_t = _apply_r(pc_t, R)
    pc_t = pc_t + t - pc
    motion_maps = mask * pc_t

    pc = pc.view(b, h, w, 3)
    pc_t = pc + motion_maps.sum(-2)
    return motion_maps, pc_t


def apply_cam_transform(pc, cam_t, cam_p, cam_r):
    b, h, w, c = pc.shape

    p = _pivot_point(cam_p)
    R = _r_mat(cam_r)

    p = p.view(b, 1, 1, 3)
    t = cam_t.view(b, 1, 1, 3)
    R = R.view(b, 1, 1, 3, 3).expand(-1, h, w, -1, -1)

    pc_t = pc - p
    pc_t = _apply_r(pc_t, R)
    pc_t = pc_t + t
    return pc_t


def optical_flow(pc, camera_intrinsics=(0.5, 0.5, 1.0)):
    points = _project_2d(pc, camera_intrinsics)
    b, h, w, c = points.shape

    x_l = np.linspace(0.0, 1.0, w)
    y_l = np.linspace(0.0, 1.0, h)
    x, y = np.meshgrid(x_l, y_l)
    pos = np.stack([x, y], -1)
    flow = points - pos
    return points, flow


def _project_2d(pc, camera_intrinsics):
    cx, cy, cf = camera_intrinsics

    X = pc[..., 0]
    Y = pc[..., 1]
    Z = pc[..., 2]

    x = cf * X / Z + cx
    y = cf * Y / Z + cy
    return torch.stack([x, y], -1)


def _pivot_point(p):
    p = p.view(-1, 20, 30)
    p_x = p.sum(1)
    p_y = p.sum(2)

    x_l = np.linspace(-30.0, 30.0, 30)
    y_l = np.linspace(-20.0, 20.0, 20)

    P_x = (p_x * x_l).sum(-1)
    P_y = (p_y * y_l).sum(-1)
    ground = np.ones_like(P_x)

    P = np.stack([P_x, P_y, ground], 1)
    return P


def _r_mat(r):
    alpha = r[:, 0] * math.pi
    beta = r[:, 1] * math.pi
    gamma = r[:, 2] * math.pi

    zero = np.zeros_like(alpha)
    one = np.ones_like(alpha)

    R_x = np.stack([
        np.stack([alpha.cos(), -alpha.sin(), zero], -1),
        np.stack([alpha.sin(), alpha.cos(), zero], -1),
        np.stack([zero, zero, one], -1),
    ], -2)

    R_y = np.stack([
        np.stack([beta.cos(), zero, beta.sin()], -1),
        np.stack([zero, one, zero], -1),
        np.stack([-beta.sin(), zero, beta.cos()], -1),
    ], -2)

    R_z = np.stack([
        np.stack([one, zero, zero], -1),
        np.stack([zero, gamma.cos(), -gamma.sin()], -1),
        np.stack([zero, gamma.sin(), gamma.cos()], -1),
    ], -2)

    return R_x @ R_y @ R_z


def _apply_r(pc, R):
    pc = pc.unsqueeze(-2)
    return (R * pc).sum(-1)

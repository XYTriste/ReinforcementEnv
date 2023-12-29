import torch
import torch.nn as nn
import torch.nn.functional as F


class BConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(BConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        return F.relu(x)


class BConv2DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding='same'):
        super(BConv2DTranspose, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        # x = self.batch_norm(x)
        return F.relu(x)


class ConvDeconvNet(nn.Module):
    def __init__(self):
        super(ConvDeconvNet, self).__init__()

        self.c11 = BConv2D(3, 32, 3)

        self.c21 = BConv2D(32, 64, 3, stride=2)
        self.c22 = BConv2D(64, 64, 3)

        self.c31 = BConv2D(64, 128, 3, stride=2)
        self.c32 = BConv2D(128, 128, 3)

        self.c41 = BConv2D(128, 256, 3, stride=2)
        self.c42 = BConv2D(256, 256, 3)

        self.c51 = BConv2D(256, 512, 3, stride=2)
        self.c52 = BConv2D(512, 512, 3)

        self.c61 = BConv2D(512, 1024, 3, stride=2)
        self.c62 = BConv2D(1024, 1024, 3)

        self.u5 = BConv2DTranspose(1024, 512, 3)
        self.u4 = BConv2DTranspose(1024, 256, 3)
        self.u3 = BConv2DTranspose(512, 128, 3)
        self.u2 = BConv2DTranspose(256, 64, 3)
        self.u1 = BConv2DTranspose(128, 32, 3)

    def forward(self, x):
        x1 = self.c11(x)

        x2 = self.c21(x1)
        x2 = self.c22(x2)

        x3 = self.c31(x2)
        x3 = self.c32(x3)

        x4 = self.c41(x3)
        x4 = self.c42(x4)

        x5 = self.c51(x4)
        x5 = self.c52(x5)

        x6 = self.c61(x5)
        embedding = self.c62(x6)

        u5 = self.u5(embedding)
        u5 = torch.cat([x5, u5], 1)

        u4 = self.u4(u5)
        u4 = torch.cat([x4, u4], 1)

        u3 = self.u3(u4)
        u3 = torch.cat([x3, u3], 1)

        u2 = self.u2(u3)
        u2 = torch.cat([x2, u2], 1)

        u1 = self.u1(u2)
        u1 = torch.cat([x1, u1], 1)
        return u1, embedding

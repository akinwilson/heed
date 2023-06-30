import torch.nn as nn
import torch.nn.functional as F

from .layers import *

import torch
import torch.nn as nn

# from torchlibrosa.stft import Spectrogram, LogmelFilterBank
# from torchlibrosa.augmentation import SpecAugmentation


class LeeNetConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size=1):

        super().__init__()
        self.pool_size = pool_size
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # print(" input x.shape")
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(" output c1 x.shape")
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(" output c2 x.shape")
        # print(x.shape)
        if self.pool_size != 1:
            x = F.max_pool1d(x, kernel_size=self.pool_size, padding=self.pool_size // 2)
            # print(" output F.max_pool1d x.shape")
            # print(x.shape)
        return x


class LeeNet(nn.Module):
    def __init__(
        self,
        dropout,
        classes_num=1,
        channel_order_dims=[1, 64, 96, 128, 256, 512, 1024],
    ):

        super().__init__()
        self.dropout = dropout
        self.classes_num = classes_num
        self.channel_order_dims = channel_order_dims
        layers = []
        for idx, c in enumerate(self.channel_order_dims):
            try:
                in_channels = c
                out_channels = self.channel_order_dims[idx + 1]
                if idx == 1:
                    pool_size = 1
                    kernel_size = 3
                    stride = 3
                    conv = LeeNetConvBlock2(
                        in_channels, out_channels, kernel_size, stride, pool_size
                    )
                else:
                    pool_size = 3
                    kernel_size = 3
                    stride = 1
                    conv = LeeNetConvBlock2(
                        in_channels, out_channels, kernel_size, stride, pool_size
                    )
                if idx != len(self.channel_order_dims):
                    layers.append(conv)
                    layers.append(nn.Dropout(p=self.dropout))
                else:
                    layers.append(conv)
            except IndexError:
                pass

        self.conv_layers = nn.Sequential(*layers)
        # self.conv_block1 = LeeNetConvBlock2(1, 64, 3, 3)
        # dim (64, )
        # self.conv_block2 = LeeNetConvBlock2(64, 96, 3, 1)
        # self.conv_block3 = LeeNetConvBlock2(96, 128, 3, 1)
        # self.conv_block4 = LeeNetConvBlock2(128, 128, 3, 1)
        # self.conv_block5 = LeeNetConvBlock2(128, 256, 3, 1)
        # self.conv_block6 = LeeNetConvBlock2(256, 256, 3, 1)
        # self.conv_block7 = LeeNetConvBlock2(256, 512, 3, 1)
        # self.conv_block8 = LeeNetConvBlock2(512, 512, 3, 1)
        # self.conv_block9 = LeeNetConvBlock2(512, 1024, 3, 1)

        # dim 45056 = (1024 * 44)
        # need to caclculate dimension size drops to  44
        self.fc1 = nn.Linear(1024 * 44, 5120 // 2, bias=True)
        self.linear = nn.Linear(5120 // 2, self.classes_num, bias=True)

    def forward(self, x):
        x = self.conv_layers(x)
        # print("x = self.conv_layers(x): ")
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        logits = self.linear(x)
        return logits

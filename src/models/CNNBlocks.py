# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        output = F.relu(output)
        return output


class ConvBatchNormReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=64, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.out_channels = growth_rate

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(in_channels + growth_rate)
        self.conv2 = nn.Conv2d(in_channels=in_channels + growth_rate, out_channels=growth_rate,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn3 = nn.BatchNorm2d(in_channels + 2 * growth_rate)
        self.conv3 = nn.Conv2d(in_channels=in_channels + 2 * growth_rate, out_channels=growth_rate,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.out_channels = in_channels + 3 * growth_rate

    def forward(self, x):
        conv1 = self.conv1(self.relu(self.bn1(x)))
        c1 = torch.cat([conv1, x], 1)

        conv2 = self.conv2(self.relu(self.bn2(c1)))
        c2 = torch.cat([c1, conv2], 1)

        conv3 = self.conv3(self.relu(self.bn3(c2)))
        c3 = torch.cat([c2, conv3], 1)

        return c3


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.out_channels = out_channels

        self.relu = nn.ReLU()

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=bias)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=bias)

    def forward(self, x):
        conv1 = self.conv1(self.relu(self.bn1(x)))
        conv2 = self.conv2(self.relu(self.bn2(conv1)))

        return conv2 + self.shortcut(self.relu(self.bn1(x)))


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        self.out_channels = out_channels * 4

        self.relu = nn.ReLU()

        self.shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1,
                                  stride=stride, bias=bias)

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel,
                      stride=stride, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, self.out_channels, kernel_size=1,
                      stride=1, bias=bias)
        )

    def forward(self, x):
        out = self.relu(self.bn1(x))

        return self.res(out) + self.shortcut(out)


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.out_features = out_features

        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.fc(x)

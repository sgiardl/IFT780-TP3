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


class BaseBlock(nn.Module):
    """
    this block is a base block for a convolutional network (Conv-BatchNorm-ReLU)
    """

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=0, bias=False):
        super(BaseBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class DenseBlock(nn.Module):
    """
    this block is a Dense Block inspired from DenseNets
    """
    def __init__(self, in_channels, growth_rate=4, stride=1, kernel_size=3, padding=1, bias=False):

        super(DenseBlock, self).__init__()

        self.bn1, self.conv1, in_channels = self.get_layers(in_channels, growth_rate, stride, kernel_size, padding, bias)
        self.bn2, self.conv2, in_channels = self.get_layers(in_channels, growth_rate, stride, kernel_size, padding, bias)
        self.bn3, self.conv3, in_channels = self.get_layers(in_channels, growth_rate, stride, kernel_size, padding, bias)

    @staticmethod
    def get_layers(in_channels, growth_rate, stride, kernel_size, padding, bias):
        bn = nn.BatchNorm2d(in_channels)
        conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        return bn, conv, in_channels+growth_rate

    def forward(self, x):

        output = torch.cat([x, self.conv1(F.relu(self.bn1(x)))], 1)
        output = torch.cat([output, self.conv2(F.relu(self.bn2(output)))], 1)
        output = torch.cat([output, self.conv3(F.relu(self.bn3(output)))], 1)

        return output


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Applies a 1x1 convolution to reduce the number of feature maps

        :param in_channels:
        :param out_channels:
        """
        super(BottleNeck, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




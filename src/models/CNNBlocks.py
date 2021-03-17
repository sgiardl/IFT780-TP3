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

class CNNBaseBlock(nn.Module):
    """
    this block is the basic block of the IFT725_NET network. it takes an
    input with in_channels, applies some blocks of batch-norm, conv and relu layers
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.relu(self.bn(self.conv(x)))
        return output
    
class DenseLayer(nn.Module):
    """
    this layer is the dense layer of the DenseBlock block.
    """
    
    def __init__(self, in_channels, growth_rate=64, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
    def forward(self, x):
        output = self.conv(self.relu(self.bn(x)))
        return output
    
class DenseBlock(nn.Module):
    """
    this block is the dense block of the IFT725_NET network.
    """
    
    def __init__(self, in_channels, growth_rate=64, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        self.Denselayer1 = DenseLayer(in_channels, growth_rate, kernel_size, stride, padding, bias)
        self.Denselayer2 = DenseLayer(in_channels+growth_rate, growth_rate, kernel_size, stride, padding, bias)
        self.Denselayer3 = DenseLayer(in_channels+(2*growth_rate), growth_rate, kernel_size, stride, padding, bias)
        
    def forward(self, x):
        denselayer1 = self.Denselayer1(x)
        cat1 = torch.cat([denselayer1, x], 1)
        
        denselayer2 = self.Denselayer2(cat1)
        cat2 = torch.cat([denselayer2, cat1], 1)
        
        denselayer3 = self.Denselayer3(cat2)
        cat3 = torch.cat([denselayer3, cat2], 1)

        return cat3
    
class ResBlock(nn.Module):
    """
    this block is the residual block of the IFT725_NET network.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)
        self.shortcut_self_bn = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(self.shortcut_conv, self.shortcut_self_bn)
        
    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        output = self.relu(output)
        return output

class BottleneckBlock(nn.Module):
    """
    this block is the bottleneck block of the IFT725_NET network.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return output

class FullyconnectedBlock(nn.Module):
    """
    this block is the Fullyconnected block of the IFT725_NET network.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        output = self.fc(x)
        return output 

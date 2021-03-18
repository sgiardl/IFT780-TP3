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
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import ConvBatchNormReluBlock, DenseBlock, ResBlock, BottleneckBlock, FullyConnectedBlock

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau IFT725UNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''


class IFT725UNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds Custom UNet model.
        Args:
            num_classes(int): number of classes.
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(IFT725UNet, self).__init__(num_classes, init_weights)

        # Encoding
        self.L1 = nn.Sequential(ConvBatchNormReluBlock(1, 64, kernel_size=5, padding=0),  # 4 x 576 x 576 -> 64 x 572 x 572
                                ResBlock(64, 64))                                         # 64 x 572 x 572 -> 64 x 572 x 572

        self.L2 = nn.Sequential(nn.MaxPool2d(2, stride=2),                                 # 64 x 572 x 572 -> 64 x 286 x 286
                                ConvBatchNormReluBlock(64, 128, kernel_size=5, padding=0), # 64 x 286 x 286 -> 128 x 282 x 282
                                ResBlock(128, 128))                                        # 128 x 282 x 282 -> 128 x 282 x 282

        self.L3 = nn.Sequential(nn.MaxPool2d(2, stride=2),                                 # 128 x 282 x 282 -> 128 x 141 x 141
                                ConvBatchNormReluBlock(128, 256, kernel_size=6, padding=0),# 128 x 141 x 141 -> 256 x 136 x 136
                                ResBlock(256, 256))                                        # 256 x 136 x 136 -> 256 x 136 x 136

        self.L4 = nn.Sequential(nn.MaxPool2d(2, stride=2),                                 # 256 x 136 x 136 -> 256 x 68 x 68
                                ConvBatchNormReluBlock(256, 512, kernel_size=5, padding=0),# 256 x 68 x 68 -> 512 x 64 x 64
                                ResBlock(512, 512))                                        # 256 x 68 x 68 -> 512 x 64 x 64

        self.Middle = nn.Sequential(nn.MaxPool2d(2, stride=2),                             # 512 x 64 x 64 -> 512 x 32 x 32
                                    ConvBatchNormReluBlock(512, 1024, kernel_size=5, padding=0), # 512 x 32 x 32 -> 1024 x 28 x 28
                                    ResBlock(1024, 1024),                                        # 1024 x 28 x 28 -> 1024 x 28 x 28
                                    ConvBatchNormReluBlock(1024, 512, kernel_size=2, padding=17))  # 1024 x 28 x 28 -> 512 x 56 x 56

        # Decoding
        self.R4 = nn.Sequential(ConvBatchNormReluBlock(1024, 512, kernel_size=5, padding=0), # 1024 x 56 x 56 -> 512 x 52 x 52
                                ResBlock(512, 512),                                          # 512 x 52 x 52 -> 512 x 52 x 52
                                ConvBatchNormReluBlock(512, 256, kernel_size=2, padding=27)) # 512 x 52 x 52 -> 256 x 104 x 104

        self.R3 = nn.Sequential(ConvBatchNormReluBlock(512, 256, kernel_size=5, padding=0),  # 512 x 104 x 104 -> 256 x 100 x 100
                                ResBlock(256, 256),                                          # 256 x 100 x 100 -> 256 x 100 x 100
                                ConvBatchNormReluBlock(256, 128, kernel_size=2, padding=51)) # 256 x 100 x 100 -> 128 x 200 x 200

        self.R2 = nn.Sequential(ConvBatchNormReluBlock(256, 128, kernel_size=5, padding=0),  # 256 x 200 x 200 -> 128 x 196 x 196
                                ResBlock(128, 128),                                          # 128 x 196 x 196 -> 128 x 196 x 196
                                ConvBatchNormReluBlock(128, 64, kernel_size=2, padding=99))  # 128 x 196 x 196 -> 64 x 392 x 392

        self.R1 = nn.Sequential(ConvBatchNormReluBlock(128, 64, kernel_size=5, padding=0),  # 128 x 392 x 392 -> 64 x 388 x 388
                                ResBlock(64, 64),                                          # 64 x 388 x 388 -> 64 x 388 x 388
                                ConvBatchNormReluBlock(64, num_classes, kernel_size=1, padding=0))  # 64 x 388 x 388 -> c x 388 x 388

        # Skip connections
        self.skip1 = nn.AdaptiveAvgPool2d(392)
        self.skip2 = nn.AdaptiveAvgPool2d(200)
        self.skip3 = nn.AdaptiveAvgPool2d(104)
        self.skip4 = nn.AdaptiveAvgPool2d(56)

    def forward(self, x):

        # Encoding
        enc1 = self.L1(x)
        enc2 = self.L2(enc1)
        enc3 = self.L3(enc2)
        enc4 = self.L4(enc3)

        # Bottom encoding -> decoding
        bottom = self.Middle(enc4)

        # Decoding
        dec4 = self.R4(torch.cat([self.skip4(enc4), bottom]))
        dec3 = self.R3(torch.cat([self.skip3(enc3), dec4]))
        dec2 = self.R2(torch.cat([self.skip2(enc2), dec3]))
        dec1 = self.R1(torch.cat([self.skip1(enc1), dec2]))

        return dec1


'''
Fin de votre code.
'''
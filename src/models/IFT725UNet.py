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
        self.L1 = nn.Sequential(ConvBatchNormReluBlock(1, 64, kernel_size=5, padding=0),  # 1 x 256 x 256 -> 64 x 252 x 252
                                ResBlock(64, 64))                                         # 64 x 252 x 252 -> 64 x 252 x 252

        self.L2 = nn.Sequential(nn.MaxPool2d(2, stride=2),                                 # 64 x 252 x 252 -> 64 x 126 x 126
                                ConvBatchNormReluBlock(64, 128, kernel_size=3, padding=0), # 64 x 126 x 126 -> 128 x 124 x 124
                                ResBlock(128, 128))                                        # 128 x 124 x 124 -> 128 x 124 x 124

        self.L3 = nn.Sequential(nn.MaxPool2d(2, stride=2),                                 # 128 x 124 x 124 -> 128 x 62 x 62
                                ConvBatchNormReluBlock(128, 256, kernel_size=3, padding=0),# 128 x 62 x 62 -> 256 x 60 x 60
                                ResBlock(256, 256))                                        # 256 x 60 x 60 -> 256 x 60 x 60

        self.L4 = nn.Sequential(nn.MaxPool2d(2, stride=2),                                 # 256 x 60 x 60 -> 256 x 30 x 30
                                ConvBatchNormReluBlock(256, 512, kernel_size=3, padding=0),# 256 x 30 x 30 -> 512 x 28 x 28
                                ResBlock(512, 512))                                        # 512 x 28 x 28 -> 512 x 28 x 28

        self.Middle = nn.Sequential(nn.MaxPool2d(2, stride=2),                             # 512 x 28 x 28 -> 512 x 14 x 14
                                    DenseBlock(512),                                       # 512 x 14 x 14 -> 704 x 14 x 14
                                    DenseBlock(704),                                       # 704 x 14 x 14 -> 896 x 14 x 14
                                    ResBlock(896, 1024),                                   # 896 x 14 x 14 -> 1024 x 14 x 14
                                    ConvBatchNormReluBlock(1024, 512, kernel_size=3, padding=8))  # 1024 x 15 x 15 -> 512 x 28 x 28

        # Decoding
        self.R4 = nn.Sequential(ConvBatchNormReluBlock(1024, 512, kernel_size=3, padding=0), # 1024 x 28 x 28 -> 512 x 26 x 26
                                ResBlock(512, 512),                                          # 512 x 26 x 26 -> 512 x 26 x 26
                                ConvBatchNormReluBlock(512, 256, kernel_size=3, padding=18)) # 512 x 26 x 26 -> 256 x 60 x 60

        self.R3 = nn.Sequential(ConvBatchNormReluBlock(512, 256, kernel_size=3, padding=0),  # 512 x 60 x 60 -> 256 x 58 x 58
                                ResBlock(256, 256),                                          # 256 x 58 x 58 -> 256 x 58 x 58
                                ConvBatchNormReluBlock(256, 128, kernel_size=3, padding=34)) # 256 x 58 x 58 -> 128 x 124 x 124

        self.R2 = nn.Sequential(ConvBatchNormReluBlock(256, 128, kernel_size=3, padding=0),  # 256 x 124 x 124 -> 128 x 122 x 122
                                ResBlock(128, 128),                                          # 128 x 122 x 122 -> 128 x 122 x 122
                                ConvBatchNormReluBlock(128, 64, kernel_size=3, padding=66))  # 128 x 122 x 122 -> 64 x 252 x 252

        self.R1 = nn.Sequential(ConvBatchNormReluBlock(128, 64, kernel_size=3, padding=0),  # 128 x 252 x 252 -> 64 x 250 x 250
                                ResBlock(64, 64),                                          # 64 x 250 x 250 -> 64 x 250 x 250
                                ConvBatchNormReluBlock(64, num_classes, kernel_size=1, padding=0))  # 64 x 250 x 250 -> c x 250 x 250

    def forward(self, x):

        # Encoding
        enc1 = self.L1(x)
        enc2 = self.L2(enc1)
        enc3 = self.L3(enc2)
        enc4 = self.L4(enc3)

        # Bottom encoding -> decoding
        bottom = self.Middle(enc4)

        # Decoding
        dec4 = self.R4(torch.cat([enc4, bottom]))
        dec3 = self.R3(torch.cat([enc3, dec4]))
        dec2 = self.R2(torch.cat([enc2, dec3]))
        dec1 = self.R1(torch.cat([enc1, dec2]))

        return dec1

'''
Fin de votre code.
'''
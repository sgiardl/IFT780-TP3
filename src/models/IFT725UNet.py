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
        Builds AlexNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super().__init__()
        
        # encoder
        in_channels = 1  # gray image
        out_channels = 64
        
        self.encoder_layer1 = torch.nn.Sequential(
            ConvBatchNormReluBlock(in_channels, 32),                          # 1 x 256 x 256 -> 32 x 256 x 256
            ResBlock(32, out_channels)                                        # 32 x 256 x 256 -> 64 x 256 x 256
        )
        self.encoder_layer2 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=2),                                      # 64 x 256 x 256 -> 64 x 128 x 128
            ConvBatchNormReluBlock(out_channels, out_channels * 2),           # 64 x 128 x 128 -> 128 x 128 x 128
            ResBlock(out_channels*2, out_channels * 2)                        # 128 x 128 x 128 -> 128 x 128 x 128
        )
        self.encoder_layer3 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=2),                                      # 128 x 128 x 128 -> 128 x 64 x 64
            ConvBatchNormReluBlock(out_channels * 2, out_channels * 4),       # 128 x 64 x 64 -> 256 x 64 x 64
            ResBlock(out_channels * 4, out_channels * 4)                      # 256 x 64 x 64 -> 256 x 64 x 64
        )
        self.encoder_layer4 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=2),                                      # 256 x 64 x 64 -> 256 x 32 x 32
            ConvBatchNormReluBlock(out_channels * 4, out_channels * 8),       # 256 x 32 x 32 -> 512 x 32 x 32
            ResBlock(out_channels * 8, out_channels * 8)                      # 512 x 32 x 32 -> 512 x 32 x 32
        )
        
        # Transitional block
        self.transitional_block = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=2),                                      # 512 x 32 x 32 -> 512 x 16 x 16
            
            ResBlock(out_channels * 8, out_channels * 8),                     # 512 x 16 x 16 -> 512 x 16 x 16
            DenseBlock(512),                                                  # 512 x 16 x 16 -> 704 x 16 x 16
            ResBlock(704, 704),                                               # 704 x 16 x 16 -> 704 x 16 x 16
            DenseBlock(704),                                                  # 704 x 16 x 16 -> 896 x 16 x 16
            ResBlock(896, 1024),                                              # 896 x 16 x 16 -> 1024 x 16 x 16

            ConvBatchNormReluBlock(out_channels * 16, out_channels * 8, padding=9)  # 1024 x 16 x 16 -> 512 x 32 x 32
        )
        
        # Decode
        self.decoder_layer4 = torch.nn.Sequential(
            ConvBatchNormReluBlock(out_channels * 16, out_channels * 4),       # 1024 x 32 x 32 -> 256 x 32 x 32
            nn.Dropout(p=0.2),
            ResBlock(out_channels * 4, out_channels * 4),                      # 256 x 32 x 32 -> 256 x 32 x 32
            ConvBatchNormReluBlock(out_channels * 4, out_channels * 4, padding=17)  # 256 x 32 x 32 -> 256 x 64 x 64
        )
        self.decoder_layer3 = torch.nn.Sequential(
            ConvBatchNormReluBlock(out_channels * 8, out_channels * 2),       # 512 x 64 x 64 -> 128 x 64 x 64
            nn.Dropout(p=0.2),
            ResBlock(out_channels * 2, out_channels * 2),                     # 128 x 64 x 64 -> 128 x 64 x 64
            ConvBatchNormReluBlock(out_channels * 2, out_channels * 2, padding=33)  # 128 x 64 x 64 -> 128 x 128 x 128
        )
        self.decoder_layer2 = torch.nn.Sequential(
            ConvBatchNormReluBlock(out_channels * 4, out_channels),           # 256 x 128 x 128 -> 64 x 128 x 128
            nn.Dropout(p=0.2),
            ResBlock(out_channels, out_channels),                             # 64 x 128 x 128 -> 64 x 128 x 128
            ConvBatchNormReluBlock(out_channels, out_channels, padding=65)        # 64 x 128 x 128 -> 64 x 256 x 256
        )
        self.decoder_layer1 = torch.nn.Sequential(
            ConvBatchNormReluBlock(out_channels * 2, 32),                    # 128 x 256 x 256 -> 32 x 256 x 256
            nn.Dropout(p=0.2),
            ResBlock(32, 32),                                                # 32 x 256 x 256 -> 32 x 256 x 256
            ConvBatchNormReluBlock(32, num_classes, kernel_size=1, padding=0)         # 32 x 256 x 256 -> num_classes x 256 x 256
        )
        
        # Skip connections
        self.skip_layer1 = nn.AdaptiveAvgPool2d(256)
        self.skip_layer2 = nn.AdaptiveAvgPool2d(128)
        self.skip_layer3 = nn.AdaptiveAvgPool2d(64)
        self.skip_layer4 = nn.AdaptiveAvgPool2d(32)

                
    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encode
        encode_1 = self.encoder_layer1(x)
        encode_2 = self.encoder_layer2(encode_1)
        encode_3 = self.encoder_layer3(encode_2)
        encode_4 = self.encoder_layer4(encode_3)

        # Transitional
        transitional = self.transitional_block(encode_4)

        # Decode
        decode_4 = self.decoder_layer4(torch.cat([self.skip_layer4(encode_4), transitional], 1))
        decode_3 = self.decoder_layer3(torch.cat([self.skip_layer3(encode_3), decode_4], 1))
        decode_2 = self.decoder_layer2(torch.cat([self.skip_layer2(encode_2), decode_3], 1))
        decode_1 = self.decoder_layer1(torch.cat([self.skip_layer1(encode_1), decode_2], 1))
        
        return decode_1

        

'''
Fin de votre code.
'''
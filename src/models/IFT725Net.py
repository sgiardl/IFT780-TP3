# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import ResidualBlock

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau IFT725Net.  Le réseau est constitué de

    1) quelques opérations de base du type « conv-batch-norm-relu »
    2) 1 (ou plus) bloc dense inspiré du modèle « denseNet)
    3) 1 (ou plus) bloc résiduel inspiré de « resNet »
    4) 1 (ou plus) bloc de couches « bottleneck » avec ou sans connexion résiduelle, c’est au choix
    5) 1 (ou plus) couches pleinement connectées

    NOTE : le code des blocks résiduels, dense et bottleneck doivent être mis dans le fichier CNNBlocks.py

'''

from models.CNNBlocks import ConvBatchNormReluBlock, DenseBlock, ResBlock, BottleneckBlock, FullyConnectedBlock


class IFT725Net(CNNBaseModel):
    """
    Class that mix up several sort of layers to create an original network
    """

    def __init__(self, num_classes=10, init_weights=True):
        """
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(IFT725Net, self).__init__(num_classes, init_weights)

        out_channels = 64
        self.output_shape = 4 * 8 * out_channels

        self.ConvBatchNormRelu1 = ConvBatchNormReluBlock(3, out_channels, stride=2)
        self.ConvBatchNormRelu2 = ConvBatchNormReluBlock(self.ConvBatchNormRelu1.out_channels, out_channels)
        self.ConvBatchNormRelu3 = ConvBatchNormReluBlock(self.ConvBatchNormRelu2.out_channels, out_channels)

        self.DenseBlock1 = DenseBlock(self.ConvBatchNormRelu3.out_channels)
        self.DenseBlock2 = DenseBlock(self.DenseBlock1.out_channels)
        self.DenseBlock3 = DenseBlock(self.DenseBlock2.out_channels)

        self.ResBlock1 = ResBlock(self.DenseBlock3.out_channels, 4 * out_channels, stride=2)
        self.ResBlock2 = ResBlock(self.ResBlock1.out_channels, 4 * out_channels)
        self.ResBlock3 = ResBlock(self.ResBlock2.out_channels, 4 * out_channels)

        self.BottleneckBlock1 = BottleneckBlock(self.ResBlock3.out_channels, 8 * out_channels, stride=2)
        self.BottleneckBlock2 = BottleneckBlock(self.BottleneckBlock1.out_channels, 8 * out_channels)
        self.BottleneckBlock3 = BottleneckBlock(self.BottleneckBlock2.out_channels, 8 * out_channels)

        self.AveragePool = nn.AdaptiveAvgPool2d(1)

        self.FullyConnectedLayer1 = FullyConnectedBlock(self.output_shape, 1024)
        self.FullyConnectedLayer2 = FullyConnectedBlock(self.FullyConnectedLayer1.out_features, num_classes)

    def forward(self, x):
        output = self.ConvBatchNormRelu1(x)
        output = self.ConvBatchNormRelu2(output)
        output = self.ConvBatchNormRelu3(output)

        output = self.DenseBlock1(output)
        output = self.DenseBlock2(output)
        output = self.DenseBlock3(output)

        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        output = self.ResBlock3(output)

        output = self.BottleneckBlock1(output)
        output = self.BottleneckBlock2(output)
        output = self.BottleneckBlock3(output)

        output = self.AveragePool(output)

        output = self.FullyConnectedLayer1(output.view(-1, self.output_shape))
        output = self.FullyConnectedLayer2(output)

        return output


'''
FIN DE VOTRE CODE
'''

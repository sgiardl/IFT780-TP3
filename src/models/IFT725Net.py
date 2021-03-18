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

        self.model = nn.Sequential(ConvBatchNormReluBlock(3, out_channels, stride=2),
                                   ConvBatchNormReluBlock(out_channels, out_channels),
                                   ConvBatchNormReluBlock(out_channels, out_channels),
                                   DenseBlock(out_channels),
                                   DenseBlock(out_channels * 4),
                                   DenseBlock(out_channels * 7),
                                   ResBlock(out_channels * 10, out_channels * 4, stride=2),
                                   ResBlock(out_channels * 4, out_channels * 4),
                                   ResBlock(out_channels * 4, out_channels * 4),
                                   BottleneckBlock(out_channels * 4, out_channels * 8, stride=2),
                                   BottleneckBlock(out_channels * 32, out_channels * 8),
                                   BottleneckBlock(out_channels * 32, out_channels * 8),
                                   nn.AdaptiveAvgPool2d(1),
                                   nn.Flatten(),
                                   FullyConnectedBlock(out_channels * 32, 1024),
                                   FullyConnectedBlock(1024, num_classes))

    def forward(self, x):
        return self.model(x)


'''
FIN DE VOTRE CODE
'''

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

from models.CNNBlocks import CNNBaseBlock
from models.CNNBlocks import DenseBlock
from models.CNNBlocks import ResBlock
from models.CNNBlocks import BottleneckBlock
from models.CNNBlocks import FullyconnectedBlock

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
        
        self.model = nn.Sequential(CNNBaseBlock(in_channels=3, out_channels=8),   # 3*32*32 ---> 8*32*32
                                   CNNBaseBlock(in_channels=8, out_channels=16),   # 8*32*32 ---> 16*32*32
                                   CNNBaseBlock(in_channels=16, out_channels=16),   # 16*32*32 ---> 16*32*32
                                   
                                   DenseBlock(in_channels=16, growth_rate=32),   # 16*32*32 ---> 96*32*32
                                   DenseBlock(in_channels=(32*3)+16, growth_rate=32),   # (32*3)*2+16*32*32 ---> 96*32*32
                                   DenseBlock(in_channels=(32*3)*2+16, growth_rate=32),   # (32*3)*3+16*32*32 ---> 96*32*32
                                   
                                   ResBlock(in_channels=16+(3*32)*3, out_channels=256, stride=2),   # (16+96)*32*32 ---> 128*16*16
                                   ResBlock(in_channels=256, out_channels=128),   # 128*32*32 ---> 64*32*32 !!!
                                   ResBlock(in_channels=128, out_channels=64),   # 64*32*32 ---> 32*32*32!!!
                                   
                                   BottleneckBlock(in_channels=64, out_channels=32, stride=2),   # 32*32*32 ---> 16*32*32!!!
                                   BottleneckBlock(in_channels=32, out_channels=16),   # 16*32*32 ---> 8*32*32!!!
                                   BottleneckBlock(in_channels=16, out_channels=8),   # 8*32*32 ---> 4*32*32!!!
                                   
                                   nn.Flatten(),   # 8*32*32 ---> (8196=8*32*32)!!!   16*8*8
                                   
                                   #FullyconnectedBlock(8*16*16, 1024),   # (8*32*32) ---> 100!!!
                                   FullyconnectedBlock(512, num_classes)   # 100 ---> 10!!!
                                  )                                   

    def forward(self, x):
        output = self.model(x)
        return output        

'''
FIN DE VOTRE CODE
'''

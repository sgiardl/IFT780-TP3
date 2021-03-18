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
        
        self.model = nn.Sequential(
                                   CNNBaseBlock(in_channels=3, out_channels=32),   # 3*32*32 ---> 32*32*32
                                   CNNBaseBlock(in_channels=32, out_channels=64),   # 32*32*32---> 64*32*32
                                   CNNBaseBlock(in_channels=64, out_channels=64, stride=2),   # 64*32*32 ---> 64*16*16

                                   DenseBlock(in_channels=64, growth_rate=32),   # 64*16*16 ---> 96+64*16*16
                                   DenseBlock(in_channels=96+64, growth_rate=32),   # 96+64*16*16---> (96*2)+64*16*16
                                   DenseBlock(in_channels=96*2+64, growth_rate=32),   # (96*2)+64*16*16---> (96*3)+64*16*16
                                   
                                   nn.Dropout(p=0.2),
                                   BottleneckBlock(in_channels=96*3+64, out_channels=96*3+64, stride=2),   # (96*3)+64*16*16 ---> (96*3)+64*8*8
                                   
                                   ResBlock(in_channels=96*3+64, out_channels=256),   # (96*3)+64*8*8 ---> 256*8*8
                                   ResBlock(in_channels=256, out_channels=128),   # 256*8*8 ---> 128*8*8
                                   ResBlock(in_channels=128, out_channels=64),   # 128*8*8 ---> 64*8*8
            
                                   nn.Dropout(p=0.1),
            
                                   CNNBaseBlock(in_channels=64, out_channels=64, stride=2),   # 64*8*8 ---> 64*4*4
                                   CNNBaseBlock(in_channels=64, out_channels=32),   # 64*4*4 ---> 32*4*4
                                   CNNBaseBlock(in_channels=32, out_channels=16),   # 32*4*4 ---> 16*4*4
                                   
                                   nn.Flatten(),   # 16*4*4 ---> 256
                                
                                   FullyconnectedBlock(256, num_classes)   # 256 ---> 10
                                  )                                   

    def forward(self, x):
        return self.model(x)        

'''
FIN DE VOTRE CODE
'''

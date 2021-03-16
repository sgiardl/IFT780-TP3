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
from models.CNNBlocks import ResidualBlock, BaseBlock, DenseBlock, BottleNeck

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

        self.model = nn.Sequential(BaseBlock(3, 10),                   # 3 x 32 x 32 -> 10 x 30 x 30
                                   DenseBlock(10),                     # 10 x 30 x 30 -> 96 x 30 x 30
                                   ResidualBlock(10+(3*32), 100),      # 106 x 30 x 30 -> 100 x 30 x 30
                                   ResidualBlock(100, 80),             # 100 x 30 x 30 -> 80 x 30 x 30
                                   ResidualBlock(80, 60),              # 80 x 30 x 30 -> 60 x 30 x 30
                                   ResidualBlock(60, 40),              # 60 x 30 x 30 -> 40 x 30 x 30
                                   BottleNeck(40, 20),                 # 40 x 30 x 30 -> 20 x 30 x 30
                                   ResidualBlock(20, 10),              # 20 x 30 x 30 -> 10 x 30 x 30
                                   BaseBlock(10, 8, kernel_size=3),    # 10 x 30 x 30 -> 8 x 28 x 28
                                   BaseBlock(8, 4, kernel_size=3),     # 8 x 28 x 28 -> 4 x 26 x 26
                                   DenseBlock(4),                      # 4 x 26 x 26 -> 100 x 26 x 26
                                   BottleNeck(100, 50),                # 100 x 26 x 26 -> 50 x 26 x 26
                                   BottleNeck(50, 25),                 # 50 x 26 x 26 -> 25 x 26 x 26
                                   BaseBlock(25, 10, kernel_size=5),   # 25 x 26 x 26 -> 10 x 22 x 22
                                   BaseBlock(10, 5, kernel_size=5),    # 10 x 22 x 22 -> 5 x 18 x 18
                                   BottleNeck(5, 1),                   # 5 x 18 x 18 -> 1 x 18 x 18
                                   nn.Flatten(),                       # 18*18 = 324
                                   nn.Linear(324, 162),                # 324 -> 162
                                   nn.Linear(162, 81),                 # 162 -> 81
                                   nn.Linear(81, num_classes))        # 100 -> num_classes

    def forward(self, x):
        return self.model(x)
'''
FIN DE VOTRE CODE
'''

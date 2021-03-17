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

        self.model = nn.Sequential(DenseBlock(3),                                 # 3 x 32 x 32 -> 99 x 32 x 32
                                   BaseBlock(99, 60, stride=2, kernel_size=4),    # 99 x 32 x 32 -> 60 x 15 x 15
                                   DenseBlock(60),                                # 60 x 15 x 15 -> 156 x 15 x 15
                                   BaseBlock(156, 100, stride=2),                 # 80 x 15 x 15 ->  100 x 7 x 7
                                   BottleNeck(100, 25),                           # 100 x 7 x 7 -> 25 x 7 x 7
                                   DenseBlock(25),                                # 25 x 7 x 7 ->  121 x 7 x 7
                                   DenseBlock(121),                               # 121 x 7 x 7 -> 217 x 7 x 7
                                   DenseBlock(217),                               # 217 x 7 x 7 -> 313 x 7 x 7
                                   BottleNeck(313, 250),                          # 293 x 7 x 7 -> 250 x 7 x 7
                                   BottleNeck(250, 200),                          # 250 x 7 x 7 -> 200 x 7 x 7
                                   ResidualBlock(200, 150),                       # 200 x 7 x 7 -> 150 x 7 x 7
                                   ResidualBlock(150, 100),                       # 150 x 7 x 7 -> 100 x 7 x 7
                                   ResidualBlock(100, 80),                        # 100 x 7 x 7 -> 80 x 7 x 7
                                   BaseBlock(80, 60),                             # 80 x 7 x 7 -> 60 x 5 x 5
                                   BaseBlock(60, 30, kernel_size=3),              # 60 x 5 x 5 -> 30 x 3 x 3
                                   BaseBlock(30, 100, kernel_size=3),             # 30 x 3 x 3 -> 100 x 1 x 1
                                   nn.Flatten(),                                  # 100 x 1 x 1 -> 100
                                   nn.Linear(100, num_classes))                   # 100 -> num_classes

    def forward(self, x):
        return self.model(x)
'''
FIN DE VOTRE CODE
'''

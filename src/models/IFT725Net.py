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
        super(IFT725Net, self).__init__()

        self.model = nn.Sequential(BaseBlock(3, 10),                   # 3 x 32 x 32 -> 10 x 30 x 30
                                   DenseBlock(10),                     # 10 x 30 x 30 -> 96 x 30 x 30
                                   ResidualBlock(10+(3*32), 100),      # 106 x 30 x 30 -> 100 x 30 x 30
                                   BottleNeck(100, 60),                # 100 x 30 x 30 -> 60 x 30 x 30
                                   ResidualBlock(60, 30),              # 60 x 30 x 30 -> 30 x 30 x 30
                                   BaseBlock(30, 15, kernel_size=5),   # 30 x 30 x 30 -> 15 x 26 x 26
                                   DenseBlock(15, growth_rate=10),     # 15 x 26 x 26 -> 45 x 26 x 26
                                   BaseBlock(45, 20, kernel_size=5),   # 45 x 26 x 26 -> 20 x 22 x 22
                                   BottleNeck(20, 5),                  # 20 x 22 x 22 -> 5 x 22 x 22
                                   BaseBlock(5, 1),                    # 5 x 22 x 22 -> 1 x 20 x 20
                                   nn.Flatten(),                       # 20*20 = 400
                                   nn.Linear(400, 100),                # 400 -> 100
                                   nn.Linear(100, num_classes))        # 100 -> num_classes

    def forward(self, x):
        return self.model(x)
'''
FIN DE VOTRE CODE
'''

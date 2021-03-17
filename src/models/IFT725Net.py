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

        self.model = nn.Sequential(BaseBlock(3, 10, stride=2, kernel_size=4),     # 3 x 32 x 32 -> 10 x 15 x 15
                                   BaseBlock(10, 5, stride=2),                    # 10 x 15 x 15 ->  5 x 7 x 7
                                   DenseBlock(5),                                 # 5 x 7 x 7 ->  101 x 7 x 7
                                   ResidualBlock(101, 80),                        # 101 x 7 x 7 -> 80 x 7 x 7
                                   BaseBlock(80, 80, padding=2),                  # 80 x 7 x 7 -> 80 x 9 x 9
                                   ResidualBlock(80, 40),                         # 80 x 9 x 9 -> 40 x 9 x 9
                                   BaseBlock(40, 40, padding=2),                  # 40 x 9 x 9 -> 40 x 11 x 11
                                   ResidualBlock(40, 20),                         # 40 x 11 x 11 -> 20 x 11 x 11
                                   DenseBlock(20, growth_rate=10),                # 20 x 11 x 11 -> 50 x 11 x 11
                                   ResidualBlock(50, 25),                         # 50 x 11 x 11 -> 25 x 11 x 11
                                   BottleNeck(25, 10),                            # 25 x 11 x 11 -> 10 x 11 x 11
                                   BaseBlock(10, 10, stride=2),                   # 10 x 11 x 11 -> 10 x 5 x 5
                                   DenseBlock(10, growth_rate=10),                # 10 x 5 x 5 -> 40 x 5 x 5
                                   ResidualBlock(40, 20),
                                   ResidualBlock(20, 10),
                                   BaseBlock(10, 10),                             # 10 x 5 x 5 -> 10 x 3 x 3,
                                   BaseBlock(10, 100),             # 10 x 4 x 4 -> 100 x 1 x 1
                                   nn.Flatten(),
                                   nn.Linear(100, num_classes))

    def forward(self, x):
        return self.model(x)
'''
FIN DE VOTRE CODE
'''

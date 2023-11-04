import torch
import torch.nn as nn

from models.blocks.vgg16_blocks import conv_block


class VGG16(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """https://arxiv.org/pdf/1409.1556.pdf"""
        super(VGG16, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # TODO: инициализируйте сверточные слои модели используя функцию conv_block
        # TODO: инициализируйте полносвязные слои модели используя nn.Linear
        raise NotImplementedError

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight)
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
        """
        raise NotImplementedError

import torch
import torch.nn as nn

from models.blocks.vgg16_blocks import conv_block, classifier_block


class VGG16(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """https://arxiv.org/pdf/1409.1556.pdf"""
        super(VGG16, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # TODO: инициализируйте сверточные слои модели, используя функцию conv_block
        self.conv1 = conv_block(...)
        self.conv2 = conv_block(...)
        ...

        # TODO: инициализируйте полносвязные слои модели, используя функцию classifier_block
        #  (последний слой инициализируется отдельно)
        self.linears = classifier_block(...)

        # TODO: инициализируйте последний полносвязный слой для классификации с помощью
        #  nn.Linear(in_features=4096, out_channels=nrof_classes)
        self.classifier = ...

        raise NotImplementedError

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight)
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           TODO: реализуйте forward pass
        """
        raise NotImplementedError

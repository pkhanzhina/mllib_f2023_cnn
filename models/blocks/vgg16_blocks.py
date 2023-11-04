import torch
import torch.nn as nn


def conv_block(in_channels: [], out_channels: [], conv_params=None, maxpool_params=None):
    """
        Функция построения одного сверточного блока нейронной сети VGG-16. Списки in_channels и out_channels задают
        последовательность сверточных слоев с соответствующими параметрами фильтров. После каждого сверточного слоя
        используется функция активации nn.RelU(inplace=True). В конце сверточных слоев необходимо применить Max Pooling

        :param in_channels: List - глубина фильтров в каждом слое
        :param out_channels: List - количество сверточных фильтров в каждом слое
        :param conv_params: None or dict - дополнительные параметры сверточных слоев
        :param maxpool_params: None or dict - параметры max pooling слоя
        :return: nn.Sequential - последовательность слоев

        # TODO: реализуйте данную функцию
    """

    assert len(in_channels) == len(out_channels)

    if conv_params is None:
        conv_params = dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if maxpool_params is None:
        maxpool_params = dict(kernel_size=2, stride=2, padding=0)

    raise NotImplementedError

import torchvision
import torch.nn as nn

from collections import OrderedDict
from convolutedbeauty.models.block import SqueezeExcitationBlock, DenseBlock


def build_se_dense():
    nn.Sequential()


class DenseNet(nn.Module):
    """
    Simple implementation of DenseNet

    link: https://arxiv.org/abs/1608.06993
    """

    def __init__(self, block, nb_channels, init_kernel_size):
        super(DenseNet, self).__init__()

        self.conv_counter = 0

        self.first_convolution(nn.Sequential(OrderedDict([
            (self._get_conv_name(), nn.Conv2d(3, )),
            ("batchnorm0", nn.BatchNorm2d()),
            ()
        ])))

    def _get_conv_name(self):
        name = "conv{}".format(self.conv_counter)

        self.conv_counter += 1

        return name

    def _build_dense_block(self, nb_blocks, in_channels, out_channels):
        blocks = [(self._get_conv_name(), self.block(in_channels, out_channels))
                  for _ in range(nb_blocks)]

        return nn.Sequential(OrderedDict(blocks))

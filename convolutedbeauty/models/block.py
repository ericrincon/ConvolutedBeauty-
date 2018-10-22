import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class SqueezeExcitationBlock(nn.Module):
    """
    Implementation of Squeeze-and-Excitation networks in a
    reusable pytorch block

    link: https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, input_size, in_channels, out_channels, reduction_ratio, name="squeezeexictation"):
        super(SqueezeExcitationBlock, self).__init__()

        self.name = name
        self.global_pooling = nn.AvgPool2d(input_size)

        # where fc refers to fully connected
        self.fc_1 = nn.Conv2d(in_channels, out_channels / reduction_ratio, input_size)
        self.fc_2 = nn.Conv2d(out_channels / reduction_ratio, in_channels, input_size)

    def forward(self, x):
        """

        :param x:
        :return:
        """

        output = self.global_pooling(x)  # 1 x 1 x C
        output = self.fc_1(output)  # 1 x 1 x C/r
        output = F.relu(output)  # 1 x 1 x C/r
        output = self.fc_2(output)  # 1 x 1 x C
        output = F.sigmoid(output)  # 1 x 1 x C

        return output


def dense_conv(in_channels, out_channels, kernel_size):
    return nn.Sequential(OrderedDict([
        ("bn", nn.BatchNorm2d()),
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size)),
        ("relu", nn.ReLU()),
    ]))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()

        self.conv1 = dense_conv(in_channels, out_channels)
        self.conv2 = dense_conv(in_channels, out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)

        return output

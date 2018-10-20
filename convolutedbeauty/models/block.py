import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitationBlock(nn.Module):
    """
    Implementation of Squeeze-and-Excitation networks in a
    reusable pytorch block

    link: https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, input_size, in_channels, out_channels, kernel_size, reduction_ratio):
        super(SqueezeExcitationBlock, self).__init__()

        self.global_pooling = nn.AvgPool2d(kernel_size)

        # where fc refers to fully connected
        self.fc_1 = nn.Conv2d(in_channels, out_channels/reduction_ratio, input_size)
        self.fc_2 = nn.Conv2d(out_channels/reduction_ratio, in_channels, input_size)

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

        return output + x

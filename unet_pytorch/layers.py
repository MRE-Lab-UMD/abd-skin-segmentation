""" Full assembly of the parts to form the complete network """

import torch.nn as nn
import torch.nn.functional as F


def double_conv_bn_relu(in_channels: int,
                        out_channels: int,
                        kernel_size: int = 3,
                        padding: int = 1):
    conv1 = nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding)
    leaky_relu_1 = nn.LeakyReLU(inplace=True)
    bn_1 = nn.BatchNorm2d(out_channels)
    conv2 = nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding)
    leaky_relu_2 = nn.LeakyReLU(inplace=True)
    bn_2 = nn.BatchNorm2d(out_channels)
    return nn.Sequential(conv1, leaky_relu_1, bn_1, conv2, leaky_relu_2, bn_2)


def deconv_double_conv_bn_relu(in_channels: int,
                               num_filters: int,
                               kernel_size: int = 3,
                               stride: int = 2):
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    double_conv = double_conv_bn_relu(in_channels, num_filters)
    return nn.Sequential(upsample, double_conv)

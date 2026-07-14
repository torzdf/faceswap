#! /usr/env/bin/python3
""" MobileNet model adapted from Keras3 version:
https://github.com/keras-team/keras/blob/v3.15.0/keras/src/applications/mobilenet.py
"""
from __future__ import annotations

import logging
import typing as T
from collections import OrderedDict

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers_legacy import Conv2dLegacy
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

_BN_EPS = 1e-3
_BN_MOM = 0.01


class DepthwiseConvBlock(nn.Module):
    """ Depthwise Convolutional Block for MobileNet

    Parameters
    ----------
    in_channels
        input channels for depthwise convolution
    out_channels
        output channels for pointwise convolution
    depth_multiplier
        The number of depthwise convolution output channels for each input channel. Default: 1
    stride
        stride for the for depthwise convolution. Default: 1
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth_multiplier: int = 1,
                 stride: int = 1) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        assert stride in (1, 2)
        depth_channels = in_channels * depth_multiplier
        self.dw = Conv2dLegacy(in_channels,
                               depth_channels,
                               3,
                               stride=stride,
                               padding="same",
                               groups=in_channels,
                               bias=False)
        self.dw_bn = nn.BatchNorm2d(depth_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.dw_act = nn.ReLU6(inplace=True)
        self.pw = nn.Conv2d(depth_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.pw_act = nn.ReLU6(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MobileNet Depthwise Convolution Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x = self.dw_act(self.dw_bn(self.dw(inputs)))  # Depthwise
        return self.pw_act(self.pw_bn(self.pw(x)))  # Pointwise


class MobileNet(nn.Module):
    """ Constructs MobileNet Model

    Reference
    ---------
    [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](
    https://arxiv.org/abs/1704.04861)

    Parameters
    ----------
    in_channels
        The number of input channels to the model. Default: 3
    classes
        The number of classes to use for classification. Default: 1000
    alpha
        Controls the width of the network. Default: 1.0
    depth_multiplier
        Depth multiplier for depthwise convolution. Default: 1
    dropout
        The dropout rate. Default: 1e-3
    """
    def __init__(self,
                 in_channels: int = 3,
                 classes: int = 1000,
                 alpha: float = 1.0,
                 depth_multiplier: int = 1,
                 dropout: float = 1e-3,) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        out_channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        out_channels = [int(x * alpha) for x in out_channels]
        strides = [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

        self.conv1 = nn.Sequential(OrderedDict([
            ("pad", nn.ZeroPad2d((0, 1, 0, 1))),
            ("conv", nn.Conv2d(in_channels,
                               out_channels[0],
                               3,
                               stride=strides[0],
                               padding=0,
                               bias=False)),
            ("bn", nn.BatchNorm2d(out_channels[0], eps=_BN_EPS, momentum=_BN_MOM)),
            ("relu", nn.ReLU6(inplace=True))
            ]))
        self.dw = nn.Sequential(*(
            DepthwiseConvBlock(in_channels * (1 if idx == 0 else depth_multiplier),
                               out_channels[idx + 1] * depth_multiplier,
                               depth_multiplier=depth_multiplier,
                               stride=stride)
            for idx, (in_channels, stride) in enumerate(zip(out_channels[:-1], strides[1:]))))
        self.classifier = nn.Sequential(OrderedDict([
            ("pool", nn.AdaptiveAvgPool2d(1)),
            ("dropout", nn.Dropout(dropout)),
            ("flatten", nn.Flatten()),
            ("fc", nn.Linear(out_channels[-1] * depth_multiplier, classes))
            ]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through MobileNet

        Parameters
        ----------
        inputs
            The input tensor to MobileNet

        Returns
        -------
        The output tensor from the MobileNet
        """
        x = inputs * 2. - 1.
        x = self.dw(self.conv1(x))
        return self.classifier(x)


def mobilenet(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> MobileNet:
    """ MobileNet model for Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    retval = MobileNet(**kwargs)
    # TODO port weights and load here
    return retval


__all__ = get_module_objects(__name__)

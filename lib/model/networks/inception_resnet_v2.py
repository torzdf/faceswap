#! /usr/env/bin/python3
""" InceptionResnet_v2 model adapted from timm:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/inception_resnet_v2.py
"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

_BN_EPS = 1e-3
_BN_MOM = 0.01


class Conv2dBn(nn.Module):
    """ Creates an Inception ResNet block

    Parameters
    ----------
    in_channels
        The number of input channels
    out_channels
        The number of output channels
    kernel_size
        The size of the convolution kernel
    stride
        The number of strides
    padding
        The amount of input padding
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


# Inception Resnet V2
class Mixed5b(nn.Module):
    """ Inception ResNet V2 Mixed 5b Block"""
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.branch0 = Conv2dBn(192, 96, 1)
        self.branch1 = nn.Sequential(Conv2dBn(192, 48, 1),
                                     Conv2dBn(48, 64, 5, padding=2))
        self.branch2 = nn.Sequential(Conv2dBn(192, 64, 1),
                                     Conv2dBn(64, 96, 3, padding=1),
                                     Conv2dBn(96, 96, 3, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
                                     Conv2dBn(192, 64, 1))

    def forward(self, inputs: torch.Tensor):
        """ Forward pass through Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x0 = self.branch0(inputs)
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        return torch.cat((x0, x1, x2, x3), 1)


class Block35(nn.Module):
    """ Inception ResNet V2 35 Block

    Parameters
    ----------
    scale
        The scaling to apply. Default: 1.0
    """
    def __init__(self, scale: float = 1.0) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.scale = scale
        self.branch0 = Conv2dBn(320, 32, 1, stride=1)
        self.branch1 = nn.Sequential(Conv2dBn(320, 32, 1, stride=1),
                                     Conv2dBn(32, 32, 3, stride=1, padding=1))
        self.branch2 = nn.Sequential(Conv2dBn(320, 32, 1, stride=1),
                                     Conv2dBn(32, 48, 3, stride=1, padding=1),
                                     Conv2dBn(48, 64, 3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, 1, stride=1)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor):
        """ Forward pass through Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x0 = self.branch0(inputs)
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + inputs
        out = self.act(out)
        return out


class Mixed6a(nn.Module):
    """ Inception ResNet V2 Mixed 6a Block"""
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.branch0 = Conv2dBn(320, 384, 3, stride=2)
        self.branch1 = nn.Sequential(Conv2dBn(320, 256, 1, stride=1),
                                     Conv2dBn(256, 256, 3, stride=1, padding=1),
                                     Conv2dBn(256, 384, 3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, inputs: torch.Tensor):
        """ Forward pass through Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x0 = self.branch0(inputs)
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    """ Inception ResNet V2 17 Block

    Parameters
    ----------
    scale
        The scaling to apply. Default: 1.0
    """
    def __init__(self, scale: float = 1.0) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.scale = scale
        self.branch0 = Conv2dBn(1088, 192, 1, stride=1)
        self.branch1 = nn.Sequential(Conv2dBn(1088, 128, 1, stride=1),
                                     Conv2dBn(128, 160, (1, 7), stride=1, padding=(0, 3)),
                                     Conv2dBn(160, 192, (7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, 1, stride=1)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor):
        """ Forward pass through Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x0 = self.branch0(inputs)
        x1 = self.branch1(inputs)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + inputs
        out = self.act(out)
        return out


class Mixed7a(nn.Module):
    """ Inception ResNet V2 Mixed 7a Block """
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.branch0 = nn.Sequential(Conv2dBn(1088, 256, 1, stride=1),
                                     Conv2dBn(256, 384, 3, stride=2))
        self.branch1 = nn.Sequential(Conv2dBn(1088, 256, 1, stride=1),
                                     Conv2dBn(256, 288, 3, stride=2))
        self.branch2 = nn.Sequential(Conv2dBn(1088, 256, 1, stride=1),
                                     Conv2dBn(256, 288, 3, stride=1, padding=1),
                                     Conv2dBn(288, 320, 3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, inputs: torch.Tensor):
        """ Forward pass through Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x0 = self.branch0(inputs)
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):
    """ Inception ResNet V2 17 Block

    Parameters
    ----------
    scale
        The scaling to apply. Default: 1.0
    no_relu
        ``True`` to skip final activation. Default: ``False``
    """
    def __init__(self,
                 scale: float = 1.0,
                 no_relu: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.scale = scale
        self.branch0 = Conv2dBn(2080, 192, 1, stride=1)
        self.branch1 = nn.Sequential(Conv2dBn(2080, 192, 1, stride=1),
                                     Conv2dBn(192, 224, (1, 3), stride=1, padding=(0, 1)),
                                     Conv2dBn(224, 256, (3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, 1, stride=1)
        self.relu = None if no_relu else nn.ReLU()

    def forward(self, inputs: torch.Tensor):
        """ Forward pass through Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x0 = self.branch0(inputs)
        x1 = self.branch1(inputs)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + inputs
        if self.relu is not None:
            out = self.relu(out)
        return out


class InceptionResnetV2(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ Inception-ResNet v2 architecture

    Reference
    ---------
    [Inception-v4, Inception-ResNet and the Impact of Residual Connections on
    Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)

    Parameters
    ----------
    num_classes
        Number of classes to classify images. Default: 1000
    in_channels
        Number of input channels. Default: 3
    global_pool
        Pooling to use prior to classifier. Default: ``avg``) -> None:
    """
    def __init__(self,
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 global_pool: T.Literal["avg", "max"] | None = "avg") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        num_features = 1536
        self.conv2d_1a = Conv2dBn(in_channels, 32, 3, stride=2)
        self.conv2d_2a = Conv2dBn(32, 32, 3, stride=1)
        self.conv2d_2b = Conv2dBn(32, 64, 3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = Conv2dBn(64, 80, 1, stride=1)
        self.conv2d_4a = Conv2dBn(80, 192, 3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed5b()
        self.repeat = nn.Sequential(*[Block35(scale=0.17) for _ in range(10)])
        self.mixed_6a = Mixed6a()
        self.repeat_1 = nn.Sequential(*[Block17(scale=0.10) for _ in range(20)])
        self.mixed_7a = Mixed7a()
        self.repeat_2 = nn.Sequential(*[Block8(scale=0.20) for _ in range(9)])
        self.block8 = Block8(no_relu=True)
        self.conv2d_7b = Conv2dBn(2080, num_features, 1, stride=1)
        self.global_pool = None
        if global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, inputs: torch.Tensor):
        """ Forward pass through Inception ResNet V2

        Parameters
        ----------
        inputs
            The input tensor to Inception ResNet V2

        Returns
        -------
        The output tensor from the Inception ResNet V2
        """
        x = inputs * 2. - 1.  # Keras compatibility
        # features
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)

        # Head
        if self.global_pool is not None:
            x = self.global_pool(x)
        return self.fc(x)


def inception_resnet_v2(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                        ) -> InceptionResnetV2:
    """ Obtain an InceptionResnetV2 model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    retval = InceptionResnetV2(**kwargs)
    # TODO port weights and load here
    return retval


__all__ = get_module_objects(__name__)

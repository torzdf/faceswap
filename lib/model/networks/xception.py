#! /usr/env/bin/python3
""" XceptionNet model adapted from timm:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/xception.py
"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn
import torch.nn.functional as F

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

_BN_EPS = 1e-3
_BN_MOM = 0.01


class SeparableConv2d(nn.Module):
    """ Separable Convolution for XceptionNet

    Parameters
    ----------
    in_channels
        The number of input channels
    out_channels
        The number of output channels
    kernel_size
        The size of the convolution kernel
    stride
        The number of strides. Default: 1
    padding
        The amount of input padding. Default: 0
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=in_channels,
                               bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

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
        return self.pointwise(self.conv1(inputs))


class Block(nn.Module):
    """ ExceptionNet Block

    Parameters
    ----------
    in_channels
        The number of input channels
    out_channels
        The number of output channels
    reps
        The number of repetitions
    stride
        The number of strides. Default: 1
    start_with_relu
        Insert ReLu activation at the start. Default: ``True``
    grow_first
        Expand at start. Default: ``True``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 reps: int,
                 strides: int = 1,
                 start_with_relu: bool = True,
                 grow_first: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOM)
                )
        else:
            self.skip = nn.Identity()

        rep = []
        for i in range(reps):
            if grow_first:
                in_c = in_channels if i == 0 else out_channels
                out_c = out_channels
            else:
                in_c = in_channels
                out_c = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_c, out_c, 3, padding=1))
            rep.append(nn.BatchNorm2d(out_c, eps=_BN_EPS, momentum=_BN_MOM))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Block

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        return self.rep(inputs) + self.skip(inputs)


class Xception(nn.Module):  # pylint: disable=too-many-instance-attributes
    """ Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf


    Parameters
    ----------
    num_classes
        The number of features to generate. Default: 1000
    in_channels
        The number of input channels to XceptionNet. Default: 3
    drop_rate
        The dropout rate. Default: 0.
    global_pool
        The pooling to use
    """
    def __init__(self,
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 drop_rate: float = 0.,
                 global_pool: T.Literal["avg", "max"] | None = "avg") -> None:
        super().__init__()
        self.drop_rate = drop_rate
        num_features = 2048

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=_BN_EPS, momentum=_BN_MOM)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=_BN_EPS, momentum=_BN_MOM)
        self.act2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)

        self.block4 = Block(728, 728, 3, 1)
        self.block5 = Block(728, 728, 3, 1)
        self.block6 = Block(728, 728, 3, 1)
        self.block7 = Block(728, 728, 3, 1)

        self.block8 = Block(728, 728, 3, 1)
        self.block9 = Block(728, 728, 3, 1)
        self.block10 = Block(728, 728, 3, 1)
        self.block11 = Block(728, 728, 3, 1)

        self.block12 = Block(728, 1024, 2, 2, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(1536, eps=_BN_EPS, momentum=_BN_MOM)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536, num_features, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features, eps=_BN_EPS, momentum=_BN_MOM)
        self.act4 = nn.ReLU(inplace=True)

        if global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the features part of the model

        Parameters
        ----------
        inputs
            The input tensor to the features

        Returns
        -------
        The output tensor from the features
        """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        return x

    def forward_head(self, inputs: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """ Forward pass through the classifier part of the model

        Parameters
        ----------
        inputs
            The input tensor to the classifier
        pre_logits
            ``True`` to skip fc layer. Default: ``False``

        Returns
        -------
        The output tensor from the classifier
        """
        x = self.global_pool(inputs)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through XceptionNet

        Parameters
        ----------
        inputs
            The input tensor to XceptionNet

        Returns
        -------
        The output tensor from XceptionNet
        """
        x = inputs * 2. - 1.
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def xception(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> Xception:
    """ Obtain an Xception model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    retval = Xception(**kwargs)
    # TODO port weights and load here
    return retval


__all__ = get_module_objects(__name__)

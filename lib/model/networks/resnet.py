#! /usr/env/bin/python3
""" ResNet models adapted directly from Keras for legacy models. TorchVision and Keras ResNet
implementations differ

https://github.com/keras-team/keras/blob/v3.15.0/keras/src/applications/resnet.py
"""
from __future__ import annotations

import logging
import typing as T
from collections import OrderedDict

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .torch_vision import load_imagenet_weights

logger = logging.getLogger(__name__)


_BN_EPS = 1.001e-5
_BN_MOM = 0.01


class ResBlockV1(nn.Module):
    """ A ResNet V1 Residual block

    Parameters
    ----------
    in_channels
        Number of input channels to the block
    out_channels
        Number of output channels in the bottleneck layer
    kernel_size
        Kernel size of the bottleneck layer. Default: 3
    stride
        Stride of the first layer. Default: 1
    conv_shortcut
        ``True`` to Use convolution shortcut, otherwise use identity shortcut. Default: ``False``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 conv_shortcut: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        if conv_shortcut:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, out_channels * 4, 1, stride=stride)),
                ("bn", nn.BatchNorm2d(out_channels * 4, eps=_BN_EPS, momentum=_BN_MOM))
            ]))
        else:
            self.shortcut = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4, eps=_BN_EPS, momentum=_BN_MOM)
        self.relu3 = nn.ReLU(inplace=True)

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
        x = self.relu1(self.bn1(self.conv1(inputs)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu3(self.shortcut(inputs) + x)


class PreActivation(nn.Module):
    """ Pre-activation layer for ResNet V2

    Parameters
    ----------
    in_channels
        Input channels to the layer
    """
    def __init__(self, in_channels: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.relu = nn.ReLU(inplace=True)

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
        return self.relu(self.bn(inputs))


class ResBlockV2(nn.Module):
    """ A ResNet V2 Residual block

    Parameters
    ----------
    in_channels
        Number of input channels to the block
    out_channels
        Number of output channels in the bottleneck layer
    kernel_size
        Kernel size of the bottleneck layer. Default: 3
    stride
        Stride of the first layer. Default: 1
    conv_shortcut
        ``True`` to Use convolution shortcut, otherwise use identity shortcut. Default: ``True``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 conv_shortcut: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._do_shortcut = conv_shortcut
        self.pre_act = PreActivation(in_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels * 4, 1, stride=stride) if conv_shortcut
            else nn.MaxPool2d(1, stride=stride) if stride > 1
            else nn.Identity()
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOM)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1)

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
        pre_act = self.pre_act(inputs)
        x = self.relu1(self.bn1(self.conv1(pre_act)))
        x = self.conv3(self.relu2(self.bn2(self.conv2(x))))
        return self.shortcut(pre_act if self._do_shortcut else inputs) + x


class ResNet(nn.Module):
    """ Torch implementation of Keras' ResNet for legacy ported models

    Parameters
    ----------
    filters
        The number of filters for each Residual Block
    blocks
        The number of blocks within each stack of Residual blocks
    version
        The ResNet version to create
    pre_activation
        ``True`` to use pre-activation (ResNet V2). Default: ``False`` (ResNet V1)
    bias
        ``True`` to use bias (ResNet V2) for convolution layers. Default: ``False`` (ResNet V1)
    include_top
        ``True`` to include the fully-connected layer at the top of the network. Default: ``True``
    classes
        Number of classes to classify images into. Default: 1000
    """
    def __init__(self,
                 filters: list[int],
                 blocks: list[int],
                 version: T.Literal[1, 2] = 1,
                 include_top: bool = True,
                 classes: int = 1000) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        if version == 1:
            self.conv1 = nn.Sequential(self.conv1, PreActivation(64))

        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        channels = [16] + filters
        self.blocks = nn.Sequential(*(self._get_res_block_stack(version,
                                                                channels[idx] * 4,
                                                                flt,
                                                                block,
                                                                stride=1 if idx == 0 else 2)
                                      for idx, (flt, block) in enumerate(zip(channels[1:],
                                                                             blocks))))
        self.post = PreActivation(channels[-1] * 4) if version == 2 else nn.Identity()
        if include_top:
            self.classifier = nn.Sequential(OrderedDict([
                ("pool", nn.AdaptiveAvgPool2d(1)),
                ("flat", nn.Flatten()),
                ("fc", nn.Linear(channels[-1] * 4, classes)),
                ("act", nn.Softmax(dim=1))
            ]))
        else:
            self.classifier = nn.Identity()

    @classmethod
    def _get_res_block_stack(cls,
                             version: T.Literal[1, 2],
                             in_channels: int,
                             out_channels: int,
                             blocks: int,
                             stride: int) -> nn.Sequential:
        """ Obtain a stack of Residual Blocks

        Parameters
        ----------
        version
            The ResNet version to obtain the stack for
        in_channels
            Number of input channels to the block
        out_channels
            Number of output channels in the bottleneck layer of a block
        blocks
            The number of blocks in the stacked blocks
        stride
            Stride of the first layer of the first block

        Returns
        -------
        Sequential module of stacked V1 Residual blocks
        """
        block = ResBlockV1 if version == 1 else ResBlockV2
        layers = [("block1", block(in_channels, out_channels, stride=stride, conv_shortcut=True))]
        for i in range(2, blocks + 1):
            layers.append((f"block{i}",
                           block(out_channels * 4,
                                 out_channels,
                                 conv_shortcut=False)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through ResNet

        Parameters
        ----------
        inputs
            The input tensor to ResNet

        Returns
        -------
        The output tensor from ResNet
        """
        x = self.pool1(self.conv1(inputs))
        x = self.post(self.blocks(x))
        return self.classifier(x)


def resnet50(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> ResNet:
    """ Obtain a ResNet 50 model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    filters = [64, 128, 256, 512]
    blocks = [3, 4, 6, 3]
    version = kwargs.pop("version", 1)
    retval = ResNet(filters, blocks, version=version, **kwargs)
    skip = None if kwargs.get("include_top", True) else ["classifier"]
    load_imagenet_weights(retval, weights, "resnet50_imagenet.pth", skip=skip)
    return retval


def resnet101(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> ResNet:
    """ Obtain a ResNet 101 model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    filters = [64, 128, 256, 512]
    blocks = [3, 4, 23, 3]
    version = kwargs.pop("version", 1)
    retval = ResNet(filters, blocks, version=version, **kwargs)
    skip = None if kwargs.get("include_top", True) else ["classifier"]
    load_imagenet_weights(retval, weights, "resnet101_imagenet.pth", skip=skip)
    return retval


def resnet152(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> ResNet:
    """ Obtain a ResNet 152 model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    filters = [64, 128, 256, 512]
    blocks = [3, 8, 36, 3]
    version = kwargs.pop("version", 1)
    retval = ResNet(filters, blocks, version=version, **kwargs)
    skip = None if kwargs.get("include_top", True) else ["classifier"]
    load_imagenet_weights(retval, weights, "resnet152_imagenet.pth", skip=skip)
    return retval


def resnet50_v2(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> ResNet:
    """ Obtain a ResNet 50 V2 model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    filters = [64, 128, 256, 512]
    blocks = [3, 4, 6, 3]
    version = kwargs.pop("version", 2)
    retval = ResNet(filters, blocks, version=version, **kwargs)
    skip = None if kwargs.get("include_top", True) else ["classifier"]
    load_imagenet_weights(retval, weights, "resnet50_v2_imagenet.pth", skip=skip)
    return retval


def resnet101_v2(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> ResNet:
    """ Obtain a ResNet V2 101 model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    filters = [64, 128, 256, 512]
    blocks = [3, 4, 23, 3]
    version = kwargs.pop("version", 2)
    retval = ResNet(filters, blocks, version=version, **kwargs)
    skip = None if kwargs.get("include_top", True) else ["classifier"]
    load_imagenet_weights(retval, weights, "resnet101_v2_imagenet.pth", skip=skip)
    return retval


def resnet152_v2(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> ResNet:
    """ Obtain a ResNet 152 V2 model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    filters = [64, 128, 256, 512]
    blocks = [3, 8, 36, 3]
    version = kwargs.pop("version", 2)
    retval = ResNet(filters, blocks, version=version, **kwargs)
    skip = None if kwargs.get("include_top", True) else ["classifier"]
    load_imagenet_weights(retval, weights, "resnet152_v2_imagenet.pth", skip=skip)
    return retval


__all__ = get_module_objects(__name__)

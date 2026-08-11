#! /usr/env/bin/python3
""" NasNet models adapted from:
https://github.com/keras-team/keras/blob/v3.15.0/keras/src/applications/nasnet.py
"""
from __future__ import annotations

import logging
import math
import typing as T
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers_legacy import Conv2dLegacy
from lib.utils import get_module_objects

from .torch_vision import load_imagenet_weights

logger = logging.getLogger(__name__)


_BN_EPS = 1e-3
_BN_MOMENTUM = 3e-4  # Value used in Keras implementation


class ConvBN(nn.Module):
    """ ConvBlock with optional ReLU on input and BatchNorm on output and no padding applied

    Parameters
    ----------
    in_channels
        The number of input channels
    out_channels
        The number of output channels
    kernel_size
        The kernel size of the convolution
    stride
        The stride of the convolution. Default: 1
    padding
        Amount of padding to apply to the convolution
    relu
        ``True`` to insert ReLU activation at the start of the block. Default: ``True``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 relu: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.relu = nn.ReLU(inplace=False) if relu else None
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM, affine=True)

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
        x = inputs if self.relu is None else self.relu(inputs)
        return self.bn(self.conv(x))


class SeparableConv2d(nn.Sequential):
    """ Separable Convolution layer for NasNetA using Tensorflow-style "same" padding. ReLU on
    input and BatchNorm on output

    Parameters
    ----------
    in_channels
        input channels to the depthwise convolution
    out_channels
        output channels for the pointwise convolution
    kernel
        Kernel size for the depthwise convolution
    stride
        Stride for the depthwise convolution. default: 1
    padding
        Padding for the depthwise convolution. default: 1
    bias
        Bias for the pointwise convolution. Default: ``False``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        if stride > 1 and padding > 0:
            self.dw = Conv2dLegacy(in_channels,
                                   in_channels,
                                   kernel,
                                   stride=stride,
                                   padding="same",
                                   bias=bias,
                                   groups=in_channels)
        else:
            self.dw = nn.Conv2d(in_channels,
                                in_channels,
                                kernel,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                                groups=in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM)


class SeparableConvBlock(nn.Sequential):
    """ 2 blocks of ReLU separable Conv-BatchNorms

    Parameters
    ----------
    in_channels
        input channels to the Separable convolutions
    out_channels
        output channels for the Separable convolutions
    kernel_size
        Kernel size for the Separable convolutions
    stride
        Stride for the Separable convolutions. Default: 1
    padding
        Padding for the Separable convolutions. Default: 1
    bias
        Bias for the Separable convolutions. Default: ``False``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.sep1 = SeparableConv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    bias=bias)
        self.sep2 = SeparableConv2d(out_channels,
                                    out_channels,
                                    kernel_size,
                                    1,
                                    padding,
                                    bias=bias)


def _pad(inputs: torch.Tensor, kernel: int, stride: int, pad_value: float = 0.0) -> torch.Tensor:
    """ Apply padding to the input Tensor using asymmetric padding if required

    Parameters
    ----------
    inputs
        The tensor to be padded
    kernel
        The size of the padding kernel
    stride
        The stride of the padding
    pad_value
        The value to pad with. Default: 0.0

    Returns
    -------
    The padded tensor
    """
    height, width = inputs.shape[-2:]
    pad_h = max((math.ceil(height / stride) - 1) * stride + kernel - height, 0)
    pad_w = max((math.ceil(width / stride) - 1) * stride + kernel - width, 0)
    return F.pad(inputs,
                 (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                 mode="constant",
                 value=pad_value)


class MaxPool(nn.Module):
    """ Padded Max Pooling layer for NasNetA that supports asymmetric padding

    Parameters
    ----------
    kernel_size
        The pooling kernel size
    stride
        The pooling stride. Default: ``None`` (The kernel size)
    """
    def __init__(self, kernel_size: int, stride: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=0)

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
        return self.pool(_pad(inputs, self._kernel_size, self._stride))


class AvgPool(nn.Module):
    """ Padded Average Pooling layer for NasNetA that supports asymmetric padding

    Parameters
    ----------
    kernel_size
        The pooling kernel size
    stride
        The pooling stride. Default: ``None`` (The kernel size)
    """
    def __init__(self, kernel_size: int, stride: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=0)

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
        return self.pool(_pad(inputs, self._kernel_size, self._stride))


class AdjustBlock(nn.Module):
    """ Adjusts the input `previous path` to match the shape of the `input`

    Parameters
    ----------
    current_channels
        The number of channels in the tensor to be reshaped
    target_channels
        The number of channels in the target to reshape to
    dims_differ
        ``True`` if the dimensional space between the 2 tensors differ. Default: ``True``
    """
    def __init__(self, current_channels: int, target_channels: int, dims_differ: bool = True
                 ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.dims_differ = dims_differ
        self.relu = nn.ReLU(inplace=False)
        if self.dims_differ:
            self.pool = nn.ModuleList((
                nn.Sequential(OrderedDict([
                    ("pool", nn.AvgPool2d(1, stride=2)),
                    ("conv", nn.Conv2d(current_channels, target_channels // 2, 1, bias=False))
                ])),
                nn.Sequential(OrderedDict([
                    ("pad", nn.ZeroPad2d((-1, 1, -1, 1))),
                    ("pool", nn.AvgPool2d(1, stride=2)),
                    ("conv", nn.Conv2d(current_channels, target_channels // 2, 1, bias=False))
                ])),
            ))
        else:
            self.conv_proj = nn.Conv2d(current_channels, target_channels, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(target_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM, affine=True)

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
        x = self.relu(inputs)
        if self.dims_differ:
            x = torch.cat([pool(x) for pool in self.pool], dim=1)
        else:
            x = self.conv_proj(x)
        return self.bn(x)


class ReductionACell(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ Reduction cell for NASNet-A

    Parameters
    ----------
    in_channels
        Number of input channels to the first convolution
    out_channels
        Number of output channels for each convolution
    prev_channels
        The number of input channels from the previous pass or ``None`` if this is the first pass
    dims_differ
        ``True`` if the dimensional space between the current and previous tensors differ.
        Default: ``True``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 prev_channels: int | None,
                 dims_differ: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        if prev_channels is not None:
            self.adjust = AdjustBlock(prev_channels, out_channels, dims_differ=dims_differ)
        self.conv1 = ConvBN(in_channels, out_channels, 1)

        in_channels_right = out_channels if prev_channels else in_channels
        self.left1 = SeparableConvBlock(out_channels, out_channels, 5, stride=2, padding=2)
        self.right1 = SeparableConvBlock(in_channels_right, out_channels, 7, stride=2, padding=3)
        self.left2 = MaxPool(3, stride=2)
        self.right2 = SeparableConvBlock(in_channels_right, out_channels, 7, stride=2, padding=3)
        self.left3 = AvgPool(3, stride=2)
        self.right3 = SeparableConvBlock(in_channels_right, out_channels, 5, stride=2, padding=2)
        self.left4 = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.left5 = SeparableConvBlock(out_channels, out_channels, 3)
        self.right5 = MaxPool(3, stride=2)

    def forward(self, inputs: torch.Tensor, inputs_prev: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the Block

        Parameters
        ----------
        inputs
            The input tensor to the block
        inputs_prev
            The previous tensor to the block. Default: ``None`` (No previous tensor)

        Returns
        -------
        output
            The output tensor
        previous
            The previous output tensor (ie: the input tensor)
        """
        x = self.conv1(inputs)
        y = inputs if inputs_prev is None else self.adjust(inputs_prev)
        x1 = self.left1(x) + self.right1(y)
        x2 = self.left2(x) + self.right2(y)
        x3 = self.left3(x) + self.right3(y)
        x4 = self.left4(x1) + x2
        x5 = self.left5(x1) + self.right5(x)
        return torch.cat([x2, x3, x4, x5], dim=1), inputs


class Stem(nn.Module):
    """ Stem for NasNetA

    Parameters
    ----------
    in_channels
        input filters
    stem_filters
        Number of filters for the stem block
    out_channels
        output filters
    filter_multiplier
        Controls the width of the network
    """
    def __init__(self,
                 in_channels: int,
                 stem_filters: int,
                 out_channels: int,
                 filter_multiplier: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        inter_channels = out_channels // (filter_multiplier ** 2)
        final_channels = out_channels // filter_multiplier
        self.out_channels_prev = inter_channels * 4
        self.out_channels = final_channels * 4

        self.conv = ConvBN(in_channels, stem_filters, 3, stride=2, relu=False)
        self.stem1 = ReductionACell(stem_filters, inter_channels, None)
        self.stem2 = ReductionACell(self.out_channels_prev,
                                    out_channels // filter_multiplier,
                                    stem_filters)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the Block

        Parameters
        ----------
        inputs
            The input tensor to the stem

        Returns
        -------
        output
            The output tensors from the stem
        previous
            The previous output tensor from the stem
        """
        x = self.conv(inputs)
        x, p = self.stem1(x)
        return self.stem2(x, p)


class NormalACell(nn.Module):
    """ Normal A Cell layer for NasNetA

    Parameters
    ----------
    in_channels
        number of input channels to the first convolution
    in_channels_prev
        number of input channels to the previous block
    out_channels
        number of output channels to each convolution
    adjust_prev_dims
        ``True`` if the previous tensor has different dimensional space
    """
    def __init__(self,
                 in_channels: int,
                 in_channels_prev: int,
                 out_channels: int,
                 adjust_prev_dims: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.adjust = AdjustBlock(in_channels_prev, out_channels, adjust_prev_dims)
        self.conv = ConvBN(in_channels, out_channels, 1)
        self.left1 = SeparableConvBlock(out_channels, out_channels, 5, padding=2)
        self.right1 = SeparableConvBlock(out_channels, out_channels, 3)
        self.left2 = SeparableConvBlock(out_channels, out_channels, 5, padding=2)
        self.right2 = SeparableConvBlock(out_channels, out_channels, 3)
        self.left3 = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.right4 = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.left4 = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.left5 = SeparableConvBlock(out_channels, out_channels, 3)

    def forward(self, inputs: torch.Tensor, inputs_prev: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Block

        Parameters
        ----------
        inputs
            The current tensor
        inputs_prev
            The previous tensor

        Returns
        -------
        The output tensor from the block
        """
        x = self.conv(inputs)
        y = self.adjust(inputs_prev)
        x0 = self.left1(x) + self.right1(y)
        x1 = self.left2(x) + self.right2(y)
        x2 = self.left3(x) + y
        x3 = self.left4(y) + self.right4(y)
        x4 = self.left5(x) + x
        return torch.cat([y, x0, x1, x2, x3, x4], 1)


class NormalA(nn.Module):
    """ The Normal cell blocks for NASNetA

    Parameters
    ----------
    in_channels
        The number of input channels to the first block
    in_channels_prev
        The number of input channels to the previous tensor
    out_channels
        The number of output channels for the first block
    num_blocks
        The number of Normal cells in the block
    """
    def __init__(self,
                 in_channels: int,
                 in_channels_prev: int,
                 out_channels: int,
                 num_blocks: int,
                 dims_differ: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        layers = []
        in_c = in_channels
        dims_differ = True
        for _ in range(num_blocks):
            layers.append(NormalACell(in_c, in_channels_prev, out_channels, dims_differ))
            in_channels_prev = in_c
            in_c = out_channels * 6
            dims_differ = False
        self.blocks = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor, inputs_prev: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the Block

        Parameters
        ----------
        inputs
            The current tensor
        inputs_prev
            The previous tensor

        Returns
        -------
        output
            The output tensor from the Normal A block
        previous
            The previous output tensor from the Normal A block
        """
        current = inputs
        prev = inputs_prev
        for block in self.blocks:
            x = block(current, prev)
            prev = current
            current = x
        return current, prev


class NASNetA(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ NASNetA Model

    Parameters
    ----------
    num_classes
        Number of classification features. Default: 1000
    stem_filters
        Number of filters in the initial stem block. Default 32
    penultimate_filters
        Number of filters in the penultimate layer. Default: 1056
    filter_multiplier
        Controls the width of the network. Default: 2
    num_blocks
        Number of repeated blocks of the NASNet model. Default: 4
    skip_reduction
        ``True`` to skip the reduction step at the tail end of the network. Default: ``False``
    include_top
        ``True`` to include the classifier output. Default: ``True``
    """
    def __init__(self,
                 num_classes: int = 1000,
                 stem_filters: int = 32,
                 penultimate_filters: int = 1056,
                 filter_multiplier: int = 2,
                 num_blocks: int = 4,
                 skip_reduction: bool = False,
                 include_top: bool = True):
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.skip_reduction = skip_reduction
        self.include_top = include_top
        filters = penultimate_filters // 24  # 24 is default value for the architecture
        self.stem = Stem(3, stem_filters, filters, filter_multiplier)
        self.normal1 = NormalA(self.stem.out_channels,
                               self.stem.out_channels_prev,
                               filters,
                               num_blocks)
        self.reduce1 = ReductionACell(6 * filters,
                                      filters * filter_multiplier,
                                      6 * filters,
                                      dims_differ=False)
        self.normal2 = NormalA(4 * filters * filter_multiplier,
                               6 * filters,
                               filters * filter_multiplier,
                               num_blocks)
        self.reduce2 = ReductionACell(6 * filters * filter_multiplier,
                                      filters * filter_multiplier ** 2,
                                      6 * filters * filter_multiplier,
                                      dims_differ=False)
        self.normal3 = NormalA(4 * filters * filter_multiplier ** 2,
                               6 * filters * filter_multiplier,
                               filters * filter_multiplier ** 2,
                               num_blocks)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        if include_top:
            self.fc = nn.Linear(24 * filters, num_classes)

    def features_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the features part of the model

        Parameters
        ----------
        inputs
            The input tensor to the features

        Returns
        -------
        The output tensor from the features
        """
        x, p = self.stem(inputs)
        x, p = self.normal1(x, p)
        x, p0 = self.reduce1(x, p)
        if not self.skip_reduction:
            p = p0
        x, p = self.normal2(x, p)
        x, p = self.reduce2(x, p)
        x, _ = self.normal3(x, p)
        return self.relu(x)

    def logits_forward(self, features: torch.Tensor) -> torch.Tensor:
        """ Obtain logits

        Parameters
        ----------
        features
            The input tensor from the features

        Returns
        -------
        The output tensor for the logits
        """
        x = T.cast(torch.Tensor, self.pool(features))
        return self.fc(self.dropout(self.flatten(x)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through NasNetAMobile

        Parameters
        ----------
        inputs
            The input tensor to NasNetAMobile

        Returns
        -------
        The output tensor from NasNetAMobile
        """
        x = self.features_forward(inputs)
        if self.include_top:
            x = self.logits_forward(x)
        return x


def nasnet_large(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> NASNetA:
    """ Obtain a NasNetA Large model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    stem_filters = kwargs.pop("stem_filters", 96)
    penultimate_filters = kwargs.pop("penultimate_filters", 4032)
    filter_multiplier = kwargs.pop("filter_multiplier", 2)
    num_blocks = kwargs.pop("num_blocks", 6)
    skip_reduction = kwargs.pop("skip_reduction", True)
    retval = NASNetA(stem_filters=stem_filters,
                     penultimate_filters=penultimate_filters,
                     filter_multiplier=filter_multiplier,
                     num_blocks=num_blocks,
                     skip_reduction=skip_reduction,
                     **kwargs)
    skip = None if kwargs.get("include_top", True) else ["fc"]
    load_imagenet_weights(retval, weights, "nasnet_large_imagenet.pth", skip=skip)
    return retval


def nasnet_mobile(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any) -> NASNetA:
    """ Obtain a NasNetA Mobile model

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    stem_filters = kwargs.pop("stem_filters", 32)
    penultimate_filters = kwargs.pop("penultimate_filters", 1056)
    filter_multiplier = kwargs.pop("filter_multiplier", 2)
    num_blocks = kwargs.pop("num_blocks", 4)
    skip_reduction = kwargs.pop("skip_reduction", False)
    retval = NASNetA(stem_filters=stem_filters,
                     penultimate_filters=penultimate_filters,
                     filter_multiplier=filter_multiplier,
                     num_blocks=num_blocks,
                     skip_reduction=skip_reduction,
                     **kwargs)
    skip = None if kwargs.get("include_top", True) else ["fc"]
    load_imagenet_weights(retval, weights, "nasnet_mobile_imagenet.pth", skip=skip)
    return retval


__all__ = get_module_objects(__name__)

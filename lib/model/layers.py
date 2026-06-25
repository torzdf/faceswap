#!/usr/bin/env python3
"""Custom Layers for faceswap.py."""
from __future__ import annotations

import logging
import math
import typing as T
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch import jit

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class InstanceNormLegacy(nn.Module):
    """A stripped down instance normalization that mimics Keras' epsilon, beta and gamma
    implementation. Don't use for new models. Use nn.InstanceNorm2d instead

    Parameters
    ----------
    eps
        Small float added to variance to avoid dividing by zero. Default: `1e-3`
    """
    def __init__(self, eps: float = 1e-3) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))  # gamma
        self.bias = nn.Parameter(torch.zeros(1))  # beta

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the legacy keras instance normalization layer

        Parameters
        ----------
        inputs
            The tensor to normalize

        Returns
        -------
            The normalized tensor
        """
        reduction_dims = tuple(range(1, len(inputs.shape)))
        mean = inputs.mean(dim=reduction_dims, keepdim=True)
        stdev = inputs.std(dim=reduction_dims, unbiased=True, keepdim=True) + self.eps
        normed = (inputs - mean) / stdev
        return normed * self.weight + self.bias


class SamePad2d(nn.Module):
    """Asymmetric padding to replicate Keras' padding='same' for backwards compatibility. This
    should not be used to new models. It exists purely to enable bit-accurate porting of Tensorflow
    built models into Torch

    Parameters
    ----------
    kernel_size
        The size of the kernel for the following convolution
    stride
        The size of the stride for the following convolution
    padding_mode
        The type of padding to apply. Default: `constant`
    """
    def __init__(self,
                 kernel_size: int,
                 stride: int,
                 mode: T.Literal["constant", "reflect"] = "constant") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.kernel = kernel_size
        self.stride = stride
        self.mode = mode

    def __repr__(self) -> str:
        """Better info for debug output"""
        return (f"{self.__class__.__name__}(kernel_size={self.kernel}, "
                f"stride={self.stride}, mode={self.mode})")

    def _pad(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform asymmetric padding to the input tensor

        Parameters
        ----------
        inputs
            The input to the padding layer

        Returns
        -------
        The padded input tensor
        """
        height, width = inputs.shape[-2:]
        pad_h = max((math.ceil(height / self.stride) - 1) * self.stride + self.kernel - height, 0)
        pad_w = max((math.ceil(width / self.stride) - 1) * self.stride + self.kernel - width, 0)
        return F.pad(inputs,
                     (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2),
                     mode=self.mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the legacy padding layer

        Parameters
        ----------
        inputs
            The tensor to be padded

        Returns
        -------
        The padded tensor
        """
        if jit.is_tracing():  # Hide nasty constant warnings when JIT tracing
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="Converting a tensor to a Python",
                                        category=jit.TracerWarning)
                return self._pad(inputs)
        return self._pad(inputs)


class SeparableConv2d(nn.Module):
    """SeparableConv2D Layer that mimics Keras' implementation in Torch

    Parameters
    ----------
    in_channels
        Number of channels in the input tensor
    out_channels
        The dimensionality of the output space (i.e. the number of filters in the pointwise
        convolution)
    kernel_size
        The size of the depthwise convolution window.
    stride
        The stride length of the depthwise convolution. strides > 1 is incompatible with
        dilation_rate > 1. Default: 1
    padding
        Padding added to all four sides of the input. Default: 0
    dilation
        The dilation rate to use for dilated convolution. Default: 1
    bias
        ``True`` if bias should be added to the output. Default: ``True``
    depth_multiplier
        The number of depthwise convolution output channels for each input channel. The total
        number of depthwise convolution output channels will be equal to
        ``input_channel * depth_multiplier``. Default: 1
    is_legacy
        ``True`` if this should use legacy padding (when kernel_size > 5 and stride > 2). For
        backwards compatibility with Keras models. Do not use this for new models.
        Default: ``False``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True,
                 depth_multiplier: int = 1,
                 is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.is_legacy = is_legacy
        if is_legacy:
            assert kernel_size > 3 and stride > 1 and padding > 0
            self.pad = SamePad2d(kernel_size, stride=stride)
            padding = 0
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels * depth_multiplier,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,  # ← one filter per input channel
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels * depth_multiplier,
                                   out_channels,
                                   kernel_size=1,
                                   bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SeparableConv2d layer

        Parameters
        ----------
        inputs
            The input to the SeparableConv2d layer

        Returns
        -------
        The output from the SeparableConv2d layer
        """
        x = inputs
        if self.is_legacy:
            x = self.pad(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


__all__ = get_module_objects(__name__)

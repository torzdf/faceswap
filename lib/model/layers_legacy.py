#!/usr/bin/env python3
""" Custom Layers that support models originally created in Keras for faceswap.py.

NOTE: None of these layers should be used for newly created models. Use the Torch versions instead
"""
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


class ConvBlockLegacy(nn.Module):
    """A standard Convolution 2D block compatible with weights from legacy Keras models

    Parameters
    ----------
    in_channels
        The number of input channels to the block
    out_channels
        The number of output channel from the block
    kernel_size
        The size of the convolution kernel
    stride
        The number of strides. Default: 1
    padding
        The padding to use. Default: "same"
    leaky_slope
        The value to use for LeakyReLu negative slope. Negative values remove activation
        altogether. Default: 0.1
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: T.Literal["same", "valid"] = "same",
                 leaky_slope: float = 0.1) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.pad = SamePad2d(kernel_size, stride) if padding == "same" else None
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0)
        self.leaky = None
        if leaky_slope >= 0.0:
            self.leaky = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Call the Faceswap Keras Convolutional Layer.

        Parameters
        ----------
        inputs
            The input to the Keras Convolutional Layer

        Returns
        -------
        The output tensor from the Keras Convolutional Layer
        """
        x = inputs
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        if self.leaky is not None:
            x = self.leaky(x)
        return x


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


class UpSampling2dLegacy(nn.Module):
    """Upsampling layer to match Keras behavior. Do not use this for new models

    Parameters
    ----------
    size
        The upsampling factors for rows and columns. Default: (2, 2)
    interpolation
        The interpolation to use. Default: "nearest"
    """
    def __init__(self,
                 size: int | tuple[int, int] = (2, 2),
                 interpolation: T.Literal["bicubic", "bilinear", "nearest"] = "nearest") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.factor = (size, size) if isinstance(size, int) else size
        self.mode = interpolation
        self.align_corners = False if interpolation in ("bicubic", "bilinear") else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Call the Upsampling2D Layer

        Parameters
        ----------
        inputs
            The input tensor to be upsampled

        Returns
        -------
        The upsampled tensor
        """
        size = (inputs.shape[-2] * self.factor[0], inputs.shape[-1] * self.factor[1])
        return F.interpolate(inputs, size=size, mode=self.mode, align_corners=self.align_corners)


__all__ = get_module_objects(__name__)

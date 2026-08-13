#!/usr/bin/env python3
""" Custom Layers that support models originally created in Keras for faceswap.py.

NOTE: None of these layers should be used for newly created models. Use the Torch versions instead
"""
from __future__ import annotations

import logging
import math
import typing as T

import torch
from torch import nn
from torch.nn import functional as F

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class InstanceNormLegacy(nn.Module):
    """ A stripped down instance normalization that mimics Keras' epsilon, beta and gamma
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
        """ Forward pass through the legacy keras instance normalization layer

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


class Conv2dLegacy(nn.Conv2d):
    """ A Convolution2D Layer that applies padding in the same method as Keras

    Provided for backwards compatibility with models created in Keras. For new plugins use
    ``nn.Conv2d``

    Parameters
    ----------
    in_channels
        Number of channels in the input image
    out_channels
        Number of channels produced by the convolution
    kernel_size
        Size of the convolving kernel
    stride
        Stride of the convolution. Default: 1
    padding
         `"valid"` means no padding. `"same"` applies TensorFlow/Keras SAME padding semantics,
         including asymmetric padding when required. Extra padding is applied to the bottom and
         right sides. When `padding="same"` and `strides=1`, the output has the same size as
         the input. Default: ``"valid"``
    dilation
        Spacing between kernel elements. Default: 1
    groups
        Number of blocked connections from input channels to output channels. Default: 1
    bias
        If ``True``, adds a learnable bias to the output. Default: ``True``
    padding_mode
        The padding mode applied by the explicit Keras-compatible padding operation.
        Default: ``"zeros"``
    """
    def __init__(self,  # pylint:disable=too-many-arguments,too-many-positional-arguments
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: T.Literal["same", "valid"] | int = "valid",
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: T.Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
                 device=None,
                 dtype=None) -> None:
        logger.debug(parse_class_init(locals()))
        assert padding in ("same", "valid"), (
            "Padding must be 'same' or 'valid'. Use nn.Conv2d for int padding")
        assert isinstance(kernel_size, int) or len(kernel_size) == 2
        assert isinstance(stride, int) or len(stride) == 2
        assert isinstance(dilation, int) or len(dilation) == 2
        self._legacy_padding = padding
        self._legacy_padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=0,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode="zeros",
                         device=device,
                         dtype=dtype)

    def pad(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply padding to the input Tensor

        Parameters
        ----------
        inputs
            The tensor to be padded

        Returns
        -------
        The padded tensor
        """
        if self._legacy_padding == "valid":
            return inputs
        height, width = inputs.shape[-2:]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation

        effective_kernel_h = (kernel_h - 1) * dilation_h + 1
        effective_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_h = max((math.ceil(height / stride_h) - 1) * stride_h + effective_kernel_h - height, 0)
        pad_w = max((math.ceil(width / stride_w) - 1) * stride_w + effective_kernel_w - width, 0)
        return F.pad(inputs,
                     (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                     mode=self._legacy_padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint:disable=redefined-builtin
        """ Forward pass through the legacy Conv2d block

        Parameters
        ----------
        input
            The input tensor to the block

        Returns
        -------
        The output tensor for the block
        """
        return super().forward(self.pad(input))


class UpSampling2dLegacy(nn.Module):
    """ Upsampling layer to match Keras behavior. Do not use this for new models

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

    def __repr__(self) -> str:
        """ Standard module repr """
        return f"{self.__class__.__name__}(size={self.factor}, interpolation={repr(self.mode)})"

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

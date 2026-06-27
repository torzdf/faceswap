#!/usr/bin/env python3
"""Neural Network Blocks for faceswap.py."""
from __future__ import annotations
import logging
import typing as T


import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects


logger = logging.getLogger(__name__)


class UpscaleSubpixel(nn.Module):
    """ An upscale layer for sub-pixel up-scaling.

    Parameters
    ----------
    in_channels
        The input channels to the upscale block
    out_channels
        The output channels from the upscale block
    kernel_size
        The kernel size to the convolution layer
    scale_factor
        The amount to upscale by image. Default: `2`
    leaky_slope
        The value to use for LeakyReLu negative slope. Negative values remove activation
        altogether. Default: 0.1.
    is_legacy
        Used to correctly pad legacy models with kernel size > 3. Should not be used for new
        models. Default: ``False``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 scale_factor: int = 2,
                 leaky_slope: float = 0.1) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.activate = leaky_slope >= 0.0
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels * scale_factor * scale_factor,
                              kernel_size,
                              stride=1,
                              padding=padding)
        if self.activate:
            self.leaky = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Call the Upscale Subpixel Layer.

        Parameters
        ----------
        inputs
            The input to the Upscale Subpixel layer

        Returns
        -------
        The output tensor from the Upscale Subpixel Layer
        """
        x = self.conv(inputs)
        if self.activate:
            x = self.leaky(x)
        return self.pixel_shuffle(x)


class ResidualBlock(nn.Module):
    """ Residual block adapted from dfaker, using legacy keras padding

    Parameters
    ----------
    channels
        The dimensionality of the input and output space (i.e. the number of input and output
        filters in the convolution)
    kernel_size
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding
        The padding to use "same", "valid" or int value. Default: "same"
    bias
        ``True`` to add learnable bias to the output. Default: ``True``
    leaky_slope
        The value to use for LeakyReLu negative slope. Default: 0.2
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: T.Literal["same", "valid"] | int = "same",
                 bias: bool = True,
                 leaky_slope: float = 0.2) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.conv1 = nn.Conv2d(channels,
                               channels,
                               kernel_size,
                               padding=padding,
                               bias=bias)
        self.leaky1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size,
                               padding=padding,
                               bias=bias)
        self.leaky2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Call the Residual Block

        Parameters
        ----------
        inputs
            The input to the Residual layer

        Returns
        -------
        The output tensor from the Residual Layer
        """
        x = self.conv1(inputs)
        x = self.leaky1(x)
        x = self.conv2(x)
        x = x + inputs
        return self.leaky2(x)


__all__ = get_module_objects(__name__)

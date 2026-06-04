#!/usr/bin/env python3
"""Neural Network Blocks for faceswap.py."""
from __future__ import annotations
import logging
import typing as T


import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from .layers import SamePad2d


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
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: T.Literal["same", "valid"] = "same") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.pad = SamePad2d(kernel_size, stride) if padding == "same" else None
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0)
        self.leaky = nn.LeakyReLU(negative_slope=0.1, inplace=True)

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
        return self.leaky(self.conv(x))


class UpscaleSubpixel(nn.Module):
    """An upscale layer for sub-pixel up-scaling.

    Parameters
    ----------
    in_channels
        The input channels to the upscale block
    out_channels
        The output channels from the upscale block
    scale_factor
        The amount to upscale by image. Default: `2`
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels * scale_factor * scale_factor,
                              3,
                              stride=1,
                              padding=1)
        self.leaky = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Call the Upscale Subpixel Layer.

        Parameters
        ----------
        inputs
            The input to the Upscale Subpixel layer

        Returns
        -------
        The output tensor from the Upscale Subpixel Layer
        """
        return self.pixel_shuffle(self.leaky(self.conv(inputs)))


__all__ = get_module_objects(__name__)

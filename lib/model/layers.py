#!/usr/bin/env python3
"""Custom Layers for faceswap.py."""
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .layers_legacy import SamePad2d, UpSampling2dLegacy

logger = logging.getLogger(__name__)


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


class Upscale2xBlock(nn.Module):
    """ Custom hybrid upscale layer for sub-pixel up-scaling.

    Most of up-scaling is approximating lighting gradients which can be accurately achieved
    using linear fitting. This layer attempts to improve memory consumption by splitting
    with bilinear and convolutional layers so that the sub-pixel update will get details
    whilst the bilinear filter will get lighting.

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Parameters
    ----------
    in_channels
        The input channels to the upscale block
    out_channels
        The output channels from the upscale block
    scale_factor
        The amount to upscale by image. Default: `2`
    sr_ratio
        The proportion of super resolution (pixel shuffler) filters to use. Non-fast mode only.
        Default: `0.5`
    fast
        Use a faster up-scaling method that may appear more rugged. Default: ``False``
    activation
        ``True`` to enable leaky_relu activation in pixel shuffler layer. Default: ``True``
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 sr_ratio: float = 0.5,
                 fast: bool = False,
                 activation: bool = True,
                 is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.fast = fast
        self.out_channels = (out_channels if fast
                             else out_channels - int(out_channels * sr_ratio))

        self.upscale = UpscaleSubpixel(in_channels,
                                       self.out_channels,
                                       scale_factor=scale_factor,
                                       leaky_slope=0.1 if activation else -1.0)
        if self.fast or (not self.fast and self.out_channels > 0):
            self.conv = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
            if is_legacy:
                self.upsample = UpSampling2dLegacy(size=scale_factor, interpolation="bilinear")
            else:
                self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

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
        x = inputs
        if self.fast:
            x = self.conv(x)
            x = self.upsample(x)
            x1 = self.upscale(inputs)
            x = x1 + x
        else:
            x_sr = self.upscale(x)
            if self.out_channels > 0:
                x = self.conv(x)
                x = self.upsample(x)
                x = torch.concat([x_sr, x], dim=1)
            else:
                x = x_sr
        return x


class UpscaleDNY(nn.Module):
    """ Upscale block that implements methodology similar to the Disney Research Paper using an
    upsampling2D block and 2 x convolutions

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    References
    ----------
    https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/

    Parameters
    ----------
    in_channels
        The input channels to the upscale block
    out_channels
        The output channels from the upscale block
    scale_factor
        The amount to upscale the image. Default: `2`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: "bilinear"
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 interpolation: T.Literal["nearest", "bilinear"] = "bilinear",
                 is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        if is_legacy:
            self.upsample = UpSampling2dLegacy(size=scale_factor, interpolation=interpolation)
        elif interpolation == "nearest":
            self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.convs = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward through the UpscaleDNY block

        Parameters
        ----------
        inputs
            The input to the block

        Returns
        -------
        The output from the block
        """
        return self.convs(self.upsample(inputs))


class UpscaleResizeImages(nn.Module):
    """ Upscale block that originally used the Keras Backend function resize_images to perform the
    up scaling, now adapted for torch. Similar in methodology to the :class:`Upscale2xBlock`

    Parameters
    ----------
    in_channels
        The input channels to the upscale block
    out_channels
        The output channels from the upscale block
    scale_factor
        The amount to upscale the image. Default: `2`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: "bilinear"
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 interpolation: T.Literal["nearest", "bilinear"] = "bilinear") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        if interpolation == "nearest":
            self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_trans = nn.ConvTranspose2d(out_channels,
                                             out_channels,
                                             3,
                                             stride=2,
                                             padding=2)  # TODO CONFIRM PADDING + IN-PADDING VS OUT_PADDING + REFLECT PADDING GETS ADDED
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Call the Faceswap Resize Images Layer.

        Parameters
        ----------
        inputs
            The input to the layer

        Returns
        -------
        The output tensor from the Upscale Layer
        """
        x_sr = self.conv(self.upsample(inputs))
        x_us = self.conv_trans(inputs)
        return self.act(x_sr + x_us)


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


__all__ = get_module_objects(__name__)

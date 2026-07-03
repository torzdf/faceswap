#!/usr/bin/env python3
""" Phaze-A Model by TorzDF with thanks to BirbFakes and the myriad of testers. """
from __future__ import annotations

import typing as T
import logging
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torchvision import models as TVMods

from lib.logger import parse_class_init
from lib.model.layers_legacy import ConvBlockLegacy, InstanceNormLegacy, UpSampling2dLegacy
from lib.model.layers import (
    AdaIN, ChannelLayerNorm, ChannelRMSNorm, GaussianNoise, Reshape, ResidualBlock, UpscaleDNY,
    Upscale2xBlock, UpscaleResizeImages, UpscaleSubpixel
    )
from lib.training.data import get_label
from lib.utils import FaceswapError, get_module_objects
from plugins.train.train_config import Loss as cfg_loss

from .base import ModelPlugin
from . import phaze_a_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code,too-many-lines

# TODO error on too small input size


UpsampleT = T.Literal["resize_images", "subpixel", "upscale_dny",
                      "upscale_fast", "upscale_hybrid", "upsample2d"]


def _get_curve(start_y: int,
               end_y: int,
               num_points: int,
               scale: float,
               mode: T.Literal["full", "cap_max", "cap_min"] = "full") -> list[int]:
    """ Obtain a curve.

    For the given start and end y values, return the y co-ordinates of a curve for the given
    number of points. The points are rounded down to the nearest 8.

    Parameters
    ----------
    start_y
        The y co-ordinate for the starting point of the curve
    end_y
        The y co-ordinate for the end point of the curve
    num_points
        The number of data points to plot on the x-axis
    scale
        The scale of the curve (from -.99 to 0.99)
    slope_mode
        The method to generate the curve. One of `"full"`, `"cap_max"` or `"cap_min"`. `"full"`
        mode generates a curve from the `"start_y"` to the `"end_y"` values. `"cap_max"` pads the
        earlier points with the `"start_y"` value before filling out the remaining points at a
        fixed divider to the `"end_y"` value. `"cap_min"` starts at the `"start_y" filling points
        at a fixed divider until the `"end_y"` value is reached and pads the remaining points with
        the `"end_y"` value. Default: `"full"`

    Returns
    -------
    List of ints of points for the given curve
    """
    scale = min(.99, max(-.99, scale))
    logger.debug("Obtaining curve: (start_y: %s, end_y: %s, num_points: %s, scale: %s, mode: %s)",
                 start_y, end_y, num_points, scale, mode)
    if mode == "full":
        x_axis = np.linspace(0., 1., num=num_points)
        y_axis: np.ndarray | list[int] = (x_axis - x_axis * scale) / (scale - abs(x_axis)
                                                                      * 2 * scale + 1)
        y_axis = T.cast(np.ndarray, y_axis) * (end_y - start_y) + start_y
        retval = [int((y // 8) * 8) for y in y_axis]
    else:
        y_axis = [start_y]
        scale = 1. - abs(scale)
        for _ in range(num_points - 1):
            current_value = max(end_y, int(((y_axis[-1] * scale) // 8) * 8))
            y_axis.append(current_value)
            if current_value == end_y:
                break
        pad = [start_y if mode == "cap_max" else end_y for _ in range(num_points - len(y_axis))]
        retval = pad + y_axis if mode == "cap_max" else y_axis + pad
    logger.debug("Returning curve: %s", retval)
    return retval


def _get_normalization(normalization: str,
                       in_channels: int,
                       is_legacy: bool) -> nn.Module:
    """ Obtain the selected normalization layer

    Parameters
    ----------
    normalization
        The normalization method to obtain
    in_channels
        The number of features that the normalization will apply to
    is_legacy
        ``True`` if the model was originally created in Keras.`

    Returns
    -------
    The configured normalization layer
    """
    # TODO other normalizations
    if is_legacy and normalization == "instance":
        retval = InstanceNormLegacy()
    elif normalization == "instance":
        retval = nn.InstanceNorm2d(in_channels, affine=True)
    elif normalization == "layer":
        retval = ChannelLayerNorm(in_channels, eps=1e-3 if is_legacy else 1e-5)
    elif normalization == "rms":
        retval = ChannelRMSNorm(in_channels, eps=1e-8 if is_legacy else None)
    else:
        raise FaceswapError(f"Invalid bottleneck_norm '{normalization}'. Choose from: "
                            "['batch', 'group', 'instance', 'layer', 'rms']")
    logger.debug("Got normalization '%s': %s", normalization, retval)
    return retval


def _scale_dim(target_resolution: int, original_dim: int) -> int:
    """ Scale a given `original_dim` so that it is a factor of the target resolution.

    Parameters
    ----------
    target_resolution
        The output resolution that is being targeted
    original_dim
        The dimension that needs to be checked for compatibility for upscaling to the
        target resolution

    Returns
    -------
    The highest dimension below or equal to `original_dim` that is a factor of the target res
    """
    new_dim = target_resolution
    while new_dim > original_dim:
        next_dim = new_dim / 2
        if not next_dim.is_integer():
            break
        new_dim = int(next_dim)
    logger.debug("target_resolution: %s, original_dim: %s, new_dim: %s",
                 target_resolution, original_dim, new_dim)
    return new_dim


def _get_upscale_layer(method: UpsampleT,
                       in_channels: int,
                       out_channels: int,
                       upsamples: int = 2,
                       is_legacy: bool = False) -> nn.Module:
    """ Obtain an instance of the requested upscale method.

    Parameters
    ----------
    method
        The user selected upscale method to use. One of `"resize_images"`, `"subpixel"`,
        `"upscale_dny"`, `"upscale_fast"`, `"upscale_hybrid"`, `"upsample2d"`
    in_channels
        The number of input channels. Used for all methods other than upsample2d
    out_channels
        The number of output channels. Used for all methods other than upsample2d
    upsamples
        The scale factor to use. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``

    Returns
    -------
    The selected configured upscale layer
    """
    if method == "upsample2d" and is_legacy:
        retval = UpSampling2dLegacy(upsamples, interpolation="bilinear")
    elif method == "upsample2d":
        retval = nn.UpsamplingBilinear2d(scale_factor=upsamples)
    elif method == "subpixel":
        retval = UpscaleSubpixel(in_channels, out_channels, scale_factor=upsamples)
    elif method == "upscale_fast":
        retval = Upscale2xBlock(in_channels,
                                out_channels,
                                scale_factor=upsamples,
                                fast=True,
                                is_legacy=is_legacy)
    elif method == "upscale_hybrid":
        retval = Upscale2xBlock(in_channels,
                                out_channels,
                                scale_factor=upsamples,
                                fast=False,
                                is_legacy=is_legacy)
    elif method == "upscale_dny":
        retval = UpscaleDNY(in_channels, out_channels, scale_factor=upsamples, is_legacy=is_legacy)
    elif method == "resize_images":  # Needs testing. May need a legacy variant for align_corners
        retval = UpscaleResizeImages(in_channels, out_channels, scale_factor=upsamples)
    else:
        raise FaceswapError(
            f"'{method}' is not a valid upscale method. Select from: ['resize_images', "
            "'subpixel', 'upscale_dny', 'upscale_fast', 'upscale_hybrid', 'upsample2d']")
    return retval


@dataclass
class _EncoderInfo:
    """ Contains model configuration options for various Phaze-A Encoders.

    Parameters
    ----------
    torch_name
        The name of the model in TorchVision. Empty string `""` if the encoder does not
        exist in TorchVision
    default_size
        The default input size of the encoder. Default: 224
    min_size
        The minimum input size that the encoder will allow. Default: 32
    kwargs
        Additional keyword arguments that can be used for building the model. Default: ``None``
    last_layer
        When the Torch model does not have a feats Sequential model, then this is the last layer
        that should be included in the encoder. All following layers are replaced with an
        nn.Identity() layer. Default: ``None`` (use feats Sequential model)
    enforce_for_weights
        ``True`` if the input size for the model must be forced to the default size when loading
        imagenet weights, otherwise ``False``. Default: ``False``
    layer_append
        Mapping of additional layers from the classifier that should be included in the model:
        (layer_name, index | None). Default: ``None`` (no additional layers)
    legacy_scaling
        The float scaling that the Keras version of the model expected. Default: `(0, 1)`
    legacy_bgr
        ``True`` if the Keras version of the model expected BGR input. Default: ``False``
    """
    torch_name: str
    default_size: int = 224
    min_size: int = 32
    kwargs: dict[str, T.Any] | None = None
    last_layer: str | None = None
    enforce_for_weights: bool = False
    layer_append: tuple[tuple[str, int | None], ...] | None = None
    legacy_scaling: tuple[int, int] = (0, 1)
    legacy_bgr: bool = False


# TODO move these builder to models?

def convnext_xlarge(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                    ) -> TVMods.convnext.ConvNeXt:
    """ ConvNext X-Large settings from Keras that does not exit in Torch"""
    block_setting = [TVMods.convnext.CNBlockConfig(256, 512, 3),
                     TVMods.convnext.CNBlockConfig(512, 1024, 3),
                     TVMods.convnext.CNBlockConfig(1024, 2048, 27),
                     TVMods.convnext.CNBlockConfig(2048, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    retval = TVMods.convnext.ConvNeXt(block_setting, stochastic_depth_prob, **kwargs)
    # TODO port weights and load here
    return retval


def efficientnet_v2_b0(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> TVMods.efficientnet.EfficientNet:
    """ EfficientNetV2_b0 settings from Keras that does not exit in Torch"""
    inverted_residual_setting = [TVMods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 1),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 2),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 48, 2),
                                 TVMods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8)]
    dropout = kwargs.pop("dropout", 0.2)
    retval = TVMods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1280, **kwargs)
    # TODO port weights and load here
    return retval


def efficientnet_v2_b1(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> TVMods.efficientnet.EfficientNet:
    """ EfficientNetV2_b1 settings from Keras that does not exit in Torch"""
    inverted_residual_setting = [TVMods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 1),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 2),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 48, 2),
                                 TVMods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8)]
    dropout = kwargs.pop("dropout", 0.2)
    retval = TVMods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1280, **kwargs
        )
    # TODO port weights and load here
    return retval


def efficientnet_v2_b2(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> TVMods.efficientnet.EfficientNet:
    """ EfficientNetV2_b2 settings from Keras that does not exit in Torch"""
    inverted_residual_setting = [TVMods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 1),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 2),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 48, 2),
                                 TVMods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8)]
    dropout = kwargs.pop("dropout", 0.2)
    retval = TVMods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1408, **kwargs
        )
    # TODO port weights and load here
    return retval


def efficientnet_v2_b3(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> TVMods.efficientnet.EfficientNet:
    """ EfficientNetV2_b3 settings from Keras that does not exit in Torch"""
    inverted_residual_setting = [TVMods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 1),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 2),
                                 TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 48, 2),
                                 TVMods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5),
                                 TVMods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8)]
    dropout = kwargs.pop("dropout", 0.2)
    retval = TVMods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1536, **kwargs
        )
    # TODO port weights and load here
    return retval


_CONVNEXT_APPEND = (("classifier", 0), )

_MODEL_MAPPING: dict[str, _EncoderInfo] = {
    "clipv_farl-b-16-16": _EncoderInfo(torch_name="FaRL-B-16-16"),  # TODO
    "clipv_farl-b-16-64": _EncoderInfo(torch_name="FaRL-B-16-64"),  # TODO
    "clipv_vit-b-16": _EncoderInfo(torch_name="ViT-B-16"),  # TODO
    "clipv_vit-b-32": _EncoderInfo(torch_name="ViT-B-32"),  # TODO
    "clipv_vit-l-14": _EncoderInfo(torch_name="ViT-L-14"),  # TODO
    "clipv_vit-l-14-336px": _EncoderInfo(torch_name="ViT-L-14-336px", default_size=336),  # TODO
    "convnext_tiny": _EncoderInfo(
        torch_name="convnext_tiny", layer_append=_CONVNEXT_APPEND, legacy_scaling=(0, 255)),
    "convnext_small": _EncoderInfo(
        torch_name="convnext_small", layer_append=_CONVNEXT_APPEND, legacy_scaling=(0, 255)),
    "convnext_base": _EncoderInfo(
            torch_name="convnext_base", layer_append=_CONVNEXT_APPEND, legacy_scaling=(0, 255)),
    "convnext_large": _EncoderInfo(
        torch_name="convnext_large", layer_append=_CONVNEXT_APPEND, legacy_scaling=(0, 255)),
    "convnext_extra_large": _EncoderInfo(
        torch_name="~convnext_xlarge", layer_append=_CONVNEXT_APPEND, legacy_scaling=(0, 255)),
    "densenet121": _EncoderInfo(torch_name="densenet121"),
    "densenet161": _EncoderInfo(torch_name="densenet161"),
    "densenet169": _EncoderInfo(torch_name="densenet169"),
    "densenet201": _EncoderInfo(torch_name="densenet201"),
    "efficientnet_b0": _EncoderInfo(torch_name="efficientnet_b0", legacy_scaling=(0, 255)),
    "efficientnet_b1": _EncoderInfo(
        torch_name="efficientnet_b1", default_size=240, legacy_scaling=(0, 255)),
    "efficientnet_b2": _EncoderInfo(
        torch_name="efficientnet_b2", default_size=260, legacy_scaling=(0, 255)),
    "efficientnet_b3": _EncoderInfo(
        torch_name="efficientnet_b3", default_size=300, legacy_scaling=(0, 255)),
    "efficientnet_b4": _EncoderInfo(
        torch_name="efficientnet_b4", default_size=380, legacy_scaling=(0, 255)),
    "efficientnet_b5": _EncoderInfo(
        torch_name="efficientnet_b5", default_size=456, legacy_scaling=(0, 255)),
    "efficientnet_b6": _EncoderInfo(
        torch_name="efficientnet_b6", default_size=528, legacy_scaling=(0, 255)),
    "efficientnet_b7": _EncoderInfo(
        torch_name="efficientnet_b7", default_size=600, legacy_scaling=(0, 255)),
    "efficientnet_v2_b0": _EncoderInfo(torch_name="~efficientnet_v2_b0", legacy_scaling=(-1, 1)),
    "efficientnet_v2_b1": _EncoderInfo(
        torch_name="~efficientnet_v2_b1", default_size=240, legacy_scaling=(-1, 1)),
    "efficientnet_v2_b2": _EncoderInfo(
        torch_name="~efficientnet_v2_b2", default_size=260, legacy_scaling=(-1, 1)),
    "efficientnet_v2_b3": _EncoderInfo(
        torch_name="~efficientnet_v2_b0", default_size=300, legacy_scaling=(-1, 1)),
    "efficientnet_v2_s": _EncoderInfo(
        torch_name="efficientnet_v2_s", default_size=384, legacy_scaling=(-1, 1)),
    "efficientnet_v2_m": _EncoderInfo(
        torch_name="efficientnet_v2_m", default_size=480, legacy_scaling=(-1, 1)),
    "efficientnet_v2_l": _EncoderInfo(
        torch_name="efficientnet_v2_l", default_size=480, legacy_scaling=(-1, 1)),
    "inception_resnet_v2": _EncoderInfo(  # TODO No Torch
        torch_name="InceptionResNetV2", min_size=75, default_size=299, legacy_scaling=(-1, 1)),
    "inception_v3": _EncoderInfo(torch_name="inception_v3",  # TODO has a flatten layer we cannot use this
                                 default_size=299,
                                 min_size=75,
                                 last_layer="Mixed_7c",
                                 legacy_scaling=(-1, 1)),
    "mobilenet": _EncoderInfo(  # TODO No Torch
        torch_name="MobileNet", legacy_scaling=(-1, 1)),
    "mobilenet_v2": _EncoderInfo(torch_name="mobilenet_v2",
                                 kwargs={"width_mult": cfg.mobilenet_width()},
                                 legacy_scaling=(-1, 1)),
    "mobilenet_v3_large": _EncoderInfo(torch_name="mobilenet_v3_large",
                                       kwargs={"width_mult": cfg.mobilenet_width(),
                                               "minimalist": cfg.mobilenet_minimalistic()},  # TODO not handled. Either remove or implement. Will need ported weights
                                       legacy_scaling=(-1, 1)),
    "mobilenet_v3_small": _EncoderInfo(torch_name="mobilenet_v3_small",
                                       kwargs={"width_mult": cfg.mobilenet_width(),
                                               "minimalist": cfg.mobilenet_minimalistic()},  # TODO not handled. Either remove or implement. Will need ported weights
                                       legacy_scaling=(-1, 1)),
    "nasnet_large": _EncoderInfo(torch_name="NASNetLarge",  # TODO No Torch
                                 default_size=331,
                                 enforce_for_weights=True,
                                 legacy_scaling=(-1, 1)),
    "nasnet_mobile": _EncoderInfo(  # TODO No Torch
        torch_name="NASNetMobile", enforce_for_weights=True, legacy_scaling=(-1, 1)),
    "resnet18": _EncoderInfo(  # TODO
        torch_name="resnet18"),
    "resnet34": _EncoderInfo(  # TODO
        torch_name="resnet34"),
    "resnet50": _EncoderInfo(  # TODO
        torch_name="resnet50", legacy_scaling=(-1, 1)),
    "resnet50_v2": _EncoderInfo(  # TODO No Torch.
        torch_name="ResNet50V2", legacy_scaling=(-1, 1)),
    "resnet101": _EncoderInfo(  # TODO
        torch_name="resnet101", legacy_scaling=(-1, 1)),
    "resnet101_v2": _EncoderInfo(  # TODO No Torch.
        torch_name="ResNet101V2", legacy_scaling=(-1, 1)),
    "resnet152": _EncoderInfo(  # TODO
        torch_name="resnet152", legacy_scaling=(-1, 1)),
    "resnet152_v2": _EncoderInfo(  # TODO No Torch
        torch_name="ResNet152V2", legacy_scaling=(-1, 1)),
    "vgg16": _EncoderInfo(torch_name="vgg16", legacy_scaling=(0, 255), legacy_bgr=True),
    "vgg19": _EncoderInfo(torch_name="vgg19", legacy_scaling=(0, 255), legacy_bgr=True),
    "xception": _EncoderInfo(  # TODO No Torch
        torch_name="Xception", min_size=71, default_size=299, legacy_scaling=(-1, 1)),
    "fs_original": _EncoderInfo(
        torch_name="", min_size=32, default_size=1024)}


class _EncoderFaceswap(nn.Module):
    """ A configurable standard Faceswap encoder based off Original model.

    Parameters
    ----------
    depth
        The number of convolutions to perform within the encoder
    min_filters
        The minimum number of filters to use for encoder convolutions. (i.e. the number of filters
        to use for the first encoder layer)
    max_filters
        The maximum number of filters to use for encoder convolutions. (i.e. the number of filters
        to use for the final encoder layer)
    input_size
        The pixel dimension of the input tensor
    use_alt
        Use a slightly alternate version of the Faceswap Encoder
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self,
                 depth: int,
                 min_filters: int,
                 max_filters: int,
                 input_size: int,
                 use_alt: bool,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        channels = [3] + [min(max_filters, min_filters * (2 ** i)) for i in range(depth)]
        self.output_shape = self._get_output_shape(input_size, depth, channels[-1], use_alt)
        """ The output shape from the encoder excluding batch dimension """
        if use_alt:
            start = channels[0]
            channels = channels[1:] + [channels[-1]]
            self.up = nn.Sequential(
                nn.Sequential(nn.Conv2d(start, channels[0], 1),
                              nn.LeakyReLU(0.2, inplace=True)),
                *(nn.Sequential(nn.Conv2d(channels[i], channels[i], 3, padding=1),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(channels[i],
                                          channels[i + 1],
                                          4 if i == depth - 1 else 3,
                                          padding=0 if i == depth - 1 else 1),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Identity() if i == depth - 1 else nn.MaxPool2d(2))
                  for i in range(depth))
            )
        elif is_legacy:
            self.up = nn.Sequential(*(
                ConvBlockLegacy(
                    channels[i], channels[i + 1], 5, stride=2, padding="same", leaky_slope=0.1
                    )
                for i in range(depth)
                ))
        else:
            self.up = nn.Sequential(*(
                nn.Sequential(nn.Conv2d(channels[i], channels[i + 1], 5, stride=2, padding=2),
                              nn.LeakyReLU(0.1, inplace=True))
                for i in range(depth)))

    @classmethod
    def _get_output_shape(cls, input_size: int, depth: int, final_channels: int, is_alt: bool
                          ) -> tuple[int, int, int]:
        """ Calculate the final output shape from the encoder """
        size = input_size
        if is_alt:
            for i in range(depth):
                if i == depth - 1:
                    size = size - 3  # final 4x4 conv
                else:
                    size = size // 2  # MaxPool2D
        else:
            for _ in range(depth):
                size = (size - 1) // 2 + 1  # k=5, stride=2, padding=2

        return (final_channels, size, size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Call the original Faceswap Encoder

        Parameters
        ----------
        inputs
            The input tensor to the Faceswap Encoder

        Returns
        -------
        The output tensor from the Faceswap Encoder
        """
        return self.up(inputs)


class Bottleneck(nn.Module):
    """ The bottleneck fully connected layer

    Parameters
    ----------
    input_shape
        The input shape to the bottleneck, excluding batch dimension
    bottleneck
        The type of bottleneck to use
    size
        The number of nodes for the dense layer (if selected)
    normalization
        The normalization method to use prior to the bottleneck
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 bottleneck: str,
                 size: int,
                 normalization: str,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        assert len(input_shape) == 3  # TODO remove after testing this holds true and then remove len check from nn.Flatten
        layers = []
        if normalization and normalization != "none":
            layers.append(_get_normalization(normalization, input_shape[0], is_legacy))
        if len(input_shape) > 1 and bottleneck in ("dense", "flatten"):
            layers.append(nn.Flatten())

        output_shape = (input_shape[0], )
        if bottleneck == "dense":
            layers.append(nn.Linear(int(np.prod(input_shape)), size))
            output_shape = (size, )
        elif bottleneck in ("average_pooling", "max_pooling"):
            lyr = nn.MaxPool2d if bottleneck == "max_pooling" else nn.AvgPool2d
            layers.append(lyr(input_shape[1]))
            layers.append(nn.Flatten())  # Flatten prior to fc layers

        self.bottleneck = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
        self.output_shape = output_shape
        """ The output shape from the bottleneck excluding batch dimension """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward through the bottleneck

        Parameters
        ----------
        inputs
            The input tensor to the Phaze-A bottleneck

        Returns
        -------
        The output tensor from the Phaze-A bottleneck
        """
        return self.bottleneck(inputs)


class Encoder(nn.Module):
    """ Encoder. Uses one of pre-existing Keras/Faceswap models or custom encoder.

    Parameters
    ----------
    architecture
        The selected encoder architecture
    input_size
        The pixel dimension of the encoder input
    bottleneck_args
        The positional args for creating the bottleneck if to be placed in the encoder or ``None``
        if it is not
    is_legacy
        ``True`` if the model was originally created in Keras. Default: ``False``
    """
    _model_kwargs = {"mobilenet": {"alpha": cfg.mobilenet_width(),
                                   "depth_multiplier": cfg.mobilenet_depth(),
                                   "dropout": cfg.mobilenet_dropout()}}
    """ Configuration option for architecture mapped to optional kwargs. """

    def __init__(self,
                 architecture: str,
                 input_size: int,
                 bottleneck_args: tuple[str, int, str] | None,
                 is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        mod_info = _MODEL_MAPPING[architecture]
        self.legacy_scaling = mod_info.legacy_scaling if is_legacy else (0, 1)
        self.encoder = self._get_encoder(mod_info, input_size, is_legacy)
        print(self.encoder)  # TODO remove
        input_size = mod_info.default_size  # TODO remove
        output_shape = T.cast(tuple[int, int, int],
                              (self._get_output_shape(input_size) if mod_info.torch_name
                               else self.encoder.output_shape))
        # TODO remove
        # for sz in range(32, 385, 16):
        #     self._get_output_shape(sz)
        # exit()
        self.bottleneck = None
        if bottleneck_args is not None:
            self.bottleneck = Bottleneck(output_shape, *bottleneck_args, is_legacy=is_legacy)
        self.output_shape = (output_shape if self.bottleneck is None
                             else self.bottleneck.output_shape)
        """ The output shape from the encoder excluding batch dimension """

    def _get_encoder(self, mod_info: _EncoderInfo, input_size: int, is_legacy: bool) -> nn.Module:
        """ Obtain the torch Module for the specified encoder architecture """
        logger.info("[Encoder] Loading encoder: '%s'", mod_info)
        if mod_info.torch_name.startswith("clipv_"):
            raise NotImplementedError  # TODO
        if mod_info.torch_name:
            is_local = mod_info.torch_name.startswith("~")
            name = mod_info.torch_name[1:] if is_local else mod_info.torch_name
            mod = sys.modules[__name__] if is_local else TVMods
            kwargs = mod_info.kwargs if mod_info.kwargs else {}
            encoder: nn.Module | nn.Sequential = getattr(mod, name)(weights="DEFAULT", **kwargs)
            if mod_info.last_layer:
                last_seen = False
                for key, _ in encoder.named_children():
                    if last_seen:
                        logger.info("[Encoder] Setting '%s' layer '%s' to nn.Identity", name, key)
                        setattr(encoder, key, nn.Identity())
                    last_seen = last_seen or key == mod_info.last_layer
                return encoder
            if not mod_info.layer_append:
                return T.cast(nn.Sequential, encoder.features)
            retval = nn.Sequential(T.cast(nn.Sequential, encoder.features))
            for layer, idx in mod_info.layer_append:
                app_layer = getattr(encoder, layer)
                if idx is not None:
                    app_layer = app_layer[idx]
                logger.info("[Encoder] Appending layer to '%s': '%s'", name, app_layer)
                retval.append(app_layer)
            return retval
        return _EncoderFaceswap(cfg.fs_original_depth(),
                                cfg.fs_original_min_filters(),
                                cfg.fs_original_max_filters(),
                                input_size,
                                cfg.fs_original_use_alt(),
                                is_legacy)

    def _get_output_shape(self, input_size: int) -> tuple[int, int, int]:
        """ Run a dummy tensor through the model to get the output shape """
        is_train = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            input_ = torch.zeros(1, 3, input_size, input_size)
            retval = tuple(self.encoder(input_).shape[1:])
        if is_train:
            self.encoder.train()
        logger.info("[Encoder] Input size: %s, Output shape: %s", input_size, retval)
        return retval

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Call the Phaze-A Encoder

        Parameters
        ----------
        inputs
            The input tensor to the Phaze-A Encoder

        Returns
        -------
        The output tensor from the Phaze-A Encoder
        """
        x = inputs
        if self.legacy_scaling == (0, 255):
            x *= 255.  # legacy model expecting inputs from 0 to 255.
        elif self.legacy_scaling == (-1, -1):
            x *= 2.  # legacy model expecting inputs from -1 to 1.
            x = x - 1.
        x = self.encoder(x)
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        return x


class UpscaleBlocks():  # pylint:disable=too-many-instance-attributes
    """ Obtain a block of upscalers.

    This class exists outside of the :class:`Decoder` model, as it is possible to place some of
    the upscalers at the end of the Fully Connected Layers, so the upscale chain needs to be able
    to be calculated by both the Fully Connected Layers and by the Decoder if required.

    For this reason, the Upscale Filter list is created as a class attribute of the
    :class:`UpscaleBlocks` layers for reference by either the Decoder or Fully Connected models

    Parameters
    ----------
    output_size
        The final pixel output size from the Phaze-A model
    min_filters
        Minimum number of filters to use for upscales
    max_filters
        Maximum number of filters to use for upscales
    slope_mode
        How to calculate the filter slope
    slope
        The filter slope scaling factor
    method
        The upscaling method to use
    gaussian
        Apply gaussian regularization at each upscale
    normalization
        Normalization to apply after each upscale
    res_blocks
        Number of res-blocks to apply to each upscale
    skip_last_residual
        ``True`` to skip the residual block for the final upscale, if residuals enabled
    learn_mask
        ``True`` if a learned mask should also be created
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self,  # pylint:disable=too-many-positional-arguments,too-many-arguments
                 output_size: int,
                 min_filters: int,
                 max_filters: int,
                 slope_mode: T.Literal["full", "cap_max", "cap_min"],
                 slope: float,
                 method: UpsampleT,
                 gaussian: bool,
                 normalization: T.Literal["none", "batch", "group", "instance", "layer", "rms"],
                 res_blocks: int,
                 skip_last_residual: bool,
                 learn_mask: bool,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._output_size = output_size
        self._filter_args = (min_filters, max_filters, slope_mode, slope)

        self._is_dny = method.lower() == "upscale_dny"
        self._method: UpsampleT = method
        self._learn_mask = learn_mask
        self._gaussian = gaussian
        self._norm_method = normalization
        self._res_blocks = res_blocks
        self._skip_last_res = skip_last_residual
        self._is_legacy = is_legacy

        self._in_channels = -1
        self._reshape_shape: tuple[int, int, int] | None = None
        self._filters: list[int] = []

    @property
    def out_channels(self) -> int:
        """ The number of filters from the final upscale layer """
        return self._filters[-1]

    def _calculate_reshape(self, input_shape: tuple[int, int, int]) -> None:
        """ Calculate whether the input needs reshaping for the chosen model output size and set to
        :attr:`_in_channels` and :attr:`_reshape_shape`

        Parameters
        ----------
        input_shape
            The shape of the Tensor feeding the Upscale Blocks (output from Inter)
        """
        old_dim = input_shape[-1]
        new_dim = _scale_dim(self._output_size, old_dim)

        if new_dim == old_dim:
            self._in_channels = input_shape[0]
        else:
            self._in_channels = int(np.prod(input_shape) // new_dim ** 2)
            self._reshape_shape = (self._in_channels, new_dim, new_dim)
        logger.debug("[UpscaleBlocks] Set in_channels: %s and reshape_shape: %s",
                     self._in_channels, self._reshape_shape)

    def _calculate_filters(self, input_shape: tuple[int, int, int]) -> None:
        """ Generate the filter curve

        Parameters
        ----------
        input_shape
            The shape of the Tensor feeding the Upscale Blocks (output from Inter)
        """
        dim = input_shape[-1] if self._reshape_shape is None else self._reshape_shape[-1]
        min_flt, max_flt, slope_mode, slope = self._filter_args
        upscales = int(np.log2(self._output_size / dim))
        self._filters = _get_curve(max_flt, min_flt, upscales, slope, mode=slope_mode)
        logger.debug("[UpscaleBlocks] Set filters: %s)", self._filters)

    def _dny_entry(self, layers: list[nn.Module]) -> None:
        """ Entry convolutions for using the upscale_dny method.

        Parameters
        ----------
        layers
            The currently building list of layers to add the dny entry convs to, if required
        """
        if not self._is_dny:
            return
        logger.debug("[UpscaleBlocks] Adding DNY entry layers")
        layers.append(nn.Sequential(
            nn.Conv2d(self._in_channels, self._filters[0], 4, padding="same"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self._filters[0], self._filters[0], 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)))

    def _add_normalization(self, layers: nn.Sequential, in_channels: int) -> None:
        """ Add a normalization layer if requested.

        Parameters
        ----------
        layers
            The sequential model building for the current upscale block
        in_channels
            The number of input channels to the normalization layer
        """
        if not self._norm_method or self._norm_method == "none":
            return

        logger.debug("[UpscaleBlocks] Adding normalization (method: '%s', in_channels: %s)",
                     self._norm_method, in_channels)
        if self._norm_method == "batch":
            layers.append(nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.01))
        elif self._norm_method == "group":
            layers.append(nn.GroupNorm(32, in_channels, eps=1e-6))  # TODO check if needs legacy
        elif self._norm_method == "instance":
            layers.append(InstanceNormLegacy() if self._is_legacy
                          else nn.InstanceNorm2d(in_channels, affine=True))
        elif self._norm_method == "layer":
            layers.append(nn.LayerNorm(in_channels, eps=1e-3))  # TODO legacy check + needs input shape, not channels
        elif self._norm_method == "rms":
            layers.append(nn.RMSNorm(in_channels, eps=1e-8))  # TODO legacy check + needs input shape, not channels
        else:
            raise FaceswapError(f"Invalid decoder_norm '{self._norm_method}'. Choose from: "
                                "['none', 'batch', 'group', 'instance', 'layer', 'rms']")

    def _upscale_block(self,
                       index: int,
                       is_mask: bool = False) -> nn.Sequential:
        """ Upscale block for Phaze-A Decoder.

        Uses requested upscale method, adds requested regularization and activation function.

        Parameters
        ----------
        index
            The filter index for the upscale
        is_mask
            ``True`` if the input is a mask. ``False`` if the input is a face. Default: ``False``

        Returns
        -------
        The sequential model for an upscale layer tensor from the upscale block
        """
        filters_ = self._filters[index]
        retval = nn.Sequential(_get_upscale_layer(self._method,
                                                  (self._in_channels if index == 0
                                                   else self._filters[index - 1]),
                                                  filters_,
                                                  upsamples=2,
                                                  is_legacy=self._is_legacy))

        if not is_mask and self._gaussian:
            retval.append(GaussianNoise(1.0))

        self._add_normalization(retval, filters_)

        skip_res = self._skip_last_res and index == len(self._filters) - 1
        if not is_mask and self._res_blocks and not skip_res:
            retval.append(nn.LeakyReLU(0.2, inplace=True))
            for _ in range(self._res_blocks):
                retval.append(ResidualBlock(filters_))
        elif not self._is_dny:  # TODO this can't be right. Missing an act on non-dny
            retval.append(nn.LeakyReLU(0.2, inplace=True))

        return retval

    def set_input_shape(self, input_shape: tuple[int, int, int]) -> None:
        """ Set the input shape to the Upscale Blocks

        Parameters
        ----------
        input_shape
            The shape of the Tensor feeding the Upscale Blocks (output from Inter)
        """
        logger.debug("[UpscaleBlocks] Setting input_shape: %s", input_shape)
        self._calculate_reshape(input_shape)
        self._calculate_filters(input_shape)

    def __call__(self, layer_indices: tuple[int, int] | None = None) -> tuple[nn.Sequential, ...]:
        """ Obtain the upscaling layers

        Parameters
        ----------
        layer_indices
            The tuple indices indicating the starting layer index and the ending layer index to
            generate upscales for. Used for when splitting upscales between the Fully Connected
            Layers and the Decoder. ``None`` will generate the full Upscale chain. An end index of
            -1 will generate the layers from the starting index to the final upscale.
            Default: ``None``

        Returns
        -------
        The upscale layers for the selected layer indices and the upscale layers for the mask
        path if learn_mask is selected
        """
        assert self._in_channels > 0 and self._filters, "Input size must be set before calling"
        start_idx, end_idx = (0, -1) if layer_indices is None else layer_indices
        end_idx = len(self._filters) if end_idx == -1 else end_idx

        layers: list[nn.Module] = []
        mask_layers: list[nn.Module] = []

        if start_idx == 0 and self._reshape_shape is not None:
            layers.append(Reshape(self._reshape_shape, is_contiguous=True))
        if start_idx == 0:
            self._dny_entry(layers)

        layers.extend(self._upscale_block(i) for i in range(start_idx, end_idx))

        if self._learn_mask:
            if start_idx == 0 and self._reshape_shape is not None:
                mask_layers.append(Reshape(self._reshape_shape, is_contiguous=True))
            if start_idx == 0:
                self._dny_entry(mask_layers)
            mask_layers.extend(self._upscale_block(i, is_mask=True)
                               for i in range(start_idx, end_idx))

        retval = [nn.Sequential(*layers)]
        if self._learn_mask:
            retval.append(nn.Sequential(*mask_layers))
        return tuple(retval)


class FullyConnected(nn.Module):
    """ Intermediate Fully Connected layers for Phaze-A Model.

    Parameters
    ----------
    input_shape
        The input shape for the fully connected layers
    feats
        List of number of features for each linear layer
    dim
        The height and width dimension for the final reshape layer at the end of the fully
        connected layers
    dropout
        Dropout amount between each FC layer
    upsampler
        The upsampler to use at the end of the FC layer
    upsamples
        How many upsamples to apply at the end of the FC layer
    upsample_filters
        The number of filters to apply to the upsampler, if required
    bottleneck_args
        The positional args for creating the bottleneck if to be placed outside the encoder or
        ``None`` if it is placed in the encoder
    upscales
        The number of decoder upscales that should be placed within the fully connected block
    upscale_blocks
        The object that builds the upscale blocks

    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self,  # pylint:disable=too-many-positional-arguments,too-many-arguments
                 input_shape: tuple[int, int, int] | tuple[int],
                 feats: list[int],
                 dim: int,
                 dropout: float,
                 upsampler: UpsampleT,
                 upsamples: int,
                 upsample_filters: int,
                 upscales: int,
                 upscale_blocks: UpscaleBlocks,
                 bottleneck_args: tuple[str, int, str] | None,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        self.dst_shape = (int(feats[-1] / (dim ** 2)), dim, dim)
        self.bottleneck = None
        if bottleneck_args is not None:
            assert len(input_shape) == 3
            self.bottleneck = Bottleneck(input_shape, *bottleneck_args, is_legacy=is_legacy)
        feats = [input_shape[0] if self.bottleneck is None
                 else self.bottleneck.output_shape[0]] + feats
        dense = []
        for i, out_feats in enumerate(feats[1:]):
            if dropout > 0.:
                dense.append(nn.Dropout(dropout, inplace=True))
            dense.append(nn.Linear(feats[i], out_feats))

        self.fc = nn.Sequential(*dense) if len(dense) > 1 else dense[0]
        self.upsamples = self._get_upsamples(upsampler,
                                             upsamples,
                                             self.dst_shape[0],
                                             upsample_filters,
                                             is_legacy)
        self.upscales = upscale_blocks((0, upscales)) if upscales else None
        self.learn_mask = self.upscales and len(self.upscales) == 2

    def _get_upsamples(self,
                       upsampler: UpsampleT,
                       upsamples: int,
                       in_channels: int,
                       out_channels: int,
                       is_legacy: bool) -> nn.Sequential | nn.Module | None:
        """ Obtain the upscale layers if requested """
        if not upsamples:
            if is_legacy and upsampler == "upsample2d":  # Bug in keras code
                return nn.LeakyReLU(0.1, inplace=True)
            return None

        if upsampler == "upsample2d" and upsamples > 1:
            retval = nn.Sequential(_get_upscale_layer(
                upsampler, in_channels, out_channels, 2 ** upsamples, is_legacy=is_legacy
                ))
        else:
            retval = nn.Sequential(*(
                _get_upscale_layer(upsampler, in_channels, out_channels, 2, is_legacy=is_legacy)
                for _ in range(upsamples)
                ))
        if upsampler == "upsample2d":
            retval.append(nn.LeakyReLU(0.1, inplace=True))
        return retval

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ Call the Phaze-A Fully Connected layer

        Parameters
        ----------
        inputs
            The input tensor to the Phaze-A Fully Connected layer

        Returns
        -------
        The output tensor from the Phaze-A Fully Connected layer. If upscales are placed within the
        FC layer and learn_mask is selected then this will be a tuple of (face, mask) output
        otherwise a single face tensor
        """
        x = inputs
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        x = T.cast(torch.Tensor, self.fc(x))
        x = x.view(x.shape[0], *self.dst_shape)
        if self.upsamples:
            x = self.upsamples(x)
        if not self.upscales:
            return x
        if not self.learn_mask:
            return self.upscales[0](x)
        return tuple(up(x) for up in self.upscales)


class Inter(nn.Module):
    """ Handles the Fully Connected layers for Phaze-A

    Parameters
    ----------
    num_identities
        The number of identities the model is training on
    input_shape
        The input shape to the FC layers
    model_output_size
        The final pixel output size from the Phaze-A model
    split_inters
        ``True`` To use separate FC layers for each side
    shared_inter
        Whether to have a shared FC layer. ``Full`` has a fully shared layer. ``Half`` places the
        shared data into identity 0's FC layer
    min_filters
        The number of 'filters' to use for the first FC layer assuming shape of (filters, dim, dim)
    max_filters
        The number of 'filters' to use for the last FC layer assuming shape of (filters, dim, dim)
    depth
        The number of fully connected layers
    slope
        The rate to move from minimum filters to maximum filters
    dim
       The dimensions to use for the final reshape layer assuming shape of (filters, dim, dim)
    dropout
        Dropout value between each FC layer
    upsampler
        Type of upsampler to use of upsamples > 0
    upsamples
        Number of upsamples to place at the end of each FC layer
    upsample_filters
        The number of filters to use for upsamplers that require them
    upscales
        The number of decoder upscales to place in the FC layers
    upscale_blocks
        The object that builds the upscale blocks
    bottleneck_args
        The positional args for creating the bottleneck if to be placed outside the encoder or
        ``None`` if it is placed in the encoder
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self,  # pylint:disable=too-many-locals,too-many-positional-arguments,too-many-arguments  # noqa[E501]
                 num_identities: int,
                 input_shape: tuple[int, int, int] | tuple[int],
                 model_output_size: int,
                 split: bool,
                 shared: T.Literal["none", "full", "half"],
                 min_filters: int,
                 max_filters: int,
                 depth: int,
                 slope: float,
                 dim: int,
                 dropout: float,
                 upsampler: UpsampleT,
                 upsamples: int,
                 upsample_filters: int,
                 upscales: int,
                 upscale_blocks: UpscaleBlocks,
                 bottleneck_args: tuple[str, int, str] | None,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        if num_identities > 2 and shared == "half":
            raise FaceswapError("half shared FC layer is not compatible with more than 2"
                                "identities")
        if not is_legacy and not split and shared != "none":
            raise FaceswapError("Shared FC layer is only compatible with split FC layers")
        super().__init__()
        # TODO legacy handling. final dim should not actually be required for filter scaling as
        # assumption was made during original implementation that upscale filters also need scaling
        # (they should not). This leads to excess filter adjustments when not necessary. Use
        # original dim instead
        final_dim = dim * (upsamples + 1)
        filters = _get_curve(
            self._scale_filters(min_filters, final_dim, model_output_size) * dim ** 2,
            self._scale_filters(max_filters, final_dim, model_output_size) * dim ** 2,
            depth,
            slope
            )

        out_filters = (upsample_filters if upsamples > 0 and upsampler != "upsample2d"
                       else filters[-1] // (final_dim * 2))

        self.shared = (None if shared == "none" else "shared" if shared == "full" else "A")
        self.output_shape = (out_filters * (2 if self.shared else 1), final_dim, final_dim)
        """ The output shape from the inter layers """
        # TODO output shape leading into G-Block may be incorrect with dec_up_in_fc or, more likely
        # we will need an FC output prior to dec upscales
        upscale_blocks.set_input_shape(self.output_shape)

        fc_args = (input_shape,
                   filters,
                   dim,
                   dropout,
                   upsampler,
                   upsamples,
                   self._scale_filters(upsample_filters, final_dim, model_output_size),  # TODO see above.
                   upscales,
                   upscale_blocks,
                   bottleneck_args,
                   is_legacy)
        if split:
            modules = {get_label(i, num_identities): FullyConnected(*fc_args)
                       for i in range(num_identities)}
        else:
            modules = {"all": FullyConnected(*fc_args)}
        if shared == "full":
            modules["shared"] = FullyConnected(*fc_args)

        self.single_inter = len(modules) == 1
        self.is_split = split
        self.fc = list(modules.values())[0] if len(modules) == 1 else nn.ModuleDict(modules)

    @classmethod
    def _scale_filters(cls, original_filters: int, final_dim: int, output_size: int) -> int:
        """ Scale the filters to be compatible with the model's selected output size.

        Parameters
        ----------
        original_filters
            The original user selected number of filters
        final_dim
            The dimensional shape of the final upsample layer
        output_size
            The final pixel output size from the Phaze-A model

        Returns
        -------
        int
            The number of filters scaled down for output size
        """
        scaled_dim = _scale_dim(output_size, final_dim)
        if scaled_dim == final_dim:
            logger.debug("filters don't require scaling. Returning: %s", original_filters)
            return original_filters

        flat = final_dim ** 2 * original_filters
        modifier = final_dim ** 2 * scaled_dim ** 2
        retval = int((flat // modifier) * modifier)
        retval = int(retval / final_dim ** 2)
        logger.debug("original_filters: %s, scaled_filters: %s", original_filters, retval)
        return retval

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """ Call the Phaze-A Intermediate layers

        Parameters
        ----------
        inputs
            The input tensors to the Phaze-A Intermediate layers for each side of the model

        Returns
        -------
        The output tensors from the Phaze-A Intermediate layers for each side of the model
        """
        if self.single_inter:  # Inverted Y or legacy error case where non-split + half shared
            return [self.fc(x) for x in inputs]

        fc = T.cast(nn.ModuleDict, self.fc)
        if not self.shared:  # Split inters no shared inter
            return [inter(x) for inter, x in zip(fc.values(), inputs)]
        if not self.is_split:  # Legacy error case where non-split + full shared
            return [torch.concat([fc["all"](enc), fc[self.shared](enc)], dim=1)
                    for enc in inputs]

        # TODO this won't work for ping-pong if/when
        return [torch.concat([fc[get_label(i, len(inputs))](enc),  # Split + shared
                              fc[self.shared](enc)], dim=1)
                for i, enc in enumerate(inputs)]


class GBlock(nn.Module):
    """ G-Block model, borrowing from Adain StyleGAN.

    Parameters
    ----------
    in_channels
        The number of channels input to the G-Block from each side's fully connected
    style_channels
        The number of nodes output from combined 'style' fc_gblock
    """
    def __init__(self, in_channels: int, style_channels: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        self.dense = nn.Sequential(
            *(nn.Sequential(nn.Linear(style_channels if i == 0 else 512, 512),
                            nn.LeakyReLU(0.1, inplace=True) if i == 2 else nn.Identity())
              for i in range(3))
            )
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.noise = GaussianNoise(1.0)

        self.g_block_recursions = 2

        # TODO this will need revisiting. Original impl created pseudo mean/std by generating 2 inp
        # I don't like it, so test new way
        # self.styles = nn.ModuleList(
        #     nn.Sequential(*(nn.Sequential(nn.Linear(in_channels, in_channels),
        #                                   Reshape((in_channels, 1, 1), is_contiguous=True))
        #                     for _ in range(2)))
        #     for _ in range(self.g_block_recursions)

        # Using in_channels * 4 just to make it work for now. This is wrong
        self.style = nn.ModuleList(nn.Sequential(nn.Linear(style_channels, in_channels * 4),
                                                 Reshape((in_channels, 2, 2), is_contiguous=True))
                                   for _ in range(self.g_block_recursions))
        self.g_noise = nn.ModuleList((nn.Sequential(GaussianNoise(1.0),
                                                    nn.Conv2d(in_channels, in_channels, 1))
                                     for _ in range(self.g_block_recursions)))
        self.g_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm = nn.ModuleList(AdaIN(dim=1) for _ in range(self.g_block_recursions))
        self.act = nn.ModuleList((nn.LeakyReLU(0.2, inplace=True)
                                  for _ in range(self.g_block_recursions)))

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the G-Block
        Parameters
        ----------
        content
            The input from the Intermediate layers
        style
            The input from the G-Block intermediate layer

        Returns
        -------
        The output tensor from the G-Block
        """
        style = self.dense(style)
        x = self.noise(self.conv(content))

        for i, (styles, noise, norm, act) in enumerate(zip(self.style,
                                                           self.g_noise,
                                                           self.norm,
                                                           self.act)):
            s = styles(style)
            n = noise(x)
            if i == self.g_block_recursions - 1:
                x = self.conv(x)
            x = norm(x, s)
            x = act(x + n)
        return x


class Decoder(nn.Module):
    """ The Decoder(s) for Phaze-A

    Parameters
    ----------
    num_identities
        The number of identities the model is training on
    split
        ``True`` if the decoders should be split. ``False`` if it is shared
    upscales_in_fc
        The number of upscales placed within the FC layers
    upscale_blocks
        The object that builds the upscale blocks
    output_kernel
        The size of the kernel for the final conv layer
    """
    def __init__(self,
                 num_identities: int,
                 split: bool,
                 upscales_in_fc: int,
                 upscale_blocks: UpscaleBlocks,
                 output_kernel: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        num_decoders = num_identities if split else 1
        indices = (upscales_in_fc, -1) if upscales_in_fc else None
        upscales = [upscale_blocks(indices) for _ in range(num_decoders)]

        self.split = split
        self.learn_mask = len(upscales[0]) == 2
        self.up = nn.ModuleList(up[0] for up in upscales)
        self.conv = nn.ModuleList(nn.Sequential(nn.Conv2d(upscale_blocks.out_channels,
                                                          3,
                                                          output_kernel,
                                                          padding=2),
                                  nn.Sigmoid())
                                  for _ in range(num_decoders))
        if self.learn_mask:
            self.mask_up = nn.ModuleList(up[1] for up in upscales)
            self.mask_conv = nn.ModuleList(nn.Sequential(nn.Conv2d(upscale_blocks.out_channels,
                                                                   1,
                                                                   output_kernel,
                                                                   padding=2),
                                                         nn.Sigmoid())
                                           for _ in range(num_decoders))

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...], ...]:
        """ Call the Phaze-A Decoder(s)

        Parameters
        ----------
        inputs
            The input tensors to the Phaze-A Decoder(s)

        Returns
        -------
        The output tensors from the Phaze-A Decoder(s)
        """
        out: list[tuple[torch.Tensor]] = []
        for idx, inp in enumerate(inputs):
            mod_idx = idx if self.split else 0
            side = [self.conv[mod_idx](self.up[mod_idx](inp))]
            if self.learn_mask:
                side.append(self.mask_conv[mod_idx](self.mask_up[mod_idx](inp)))
            out.append(tuple(side))
        return tuple(out)


# TODO make sure dropouts are done (needed custom handling in keras)
class PhazeA(ModelPlugin):
    """ Phaze-A Faceswap Model.

    An highly adaptable and configurable model by torzDF

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, num_identities: int = 2, is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        if cfg.output_size() % 16 != 0:
            raise FaceswapError("Phaze-A output shape must be a multiple of 16")
        self._validate_encoder_architecture()
        input_size = self._get_input_size()
        is_bgr = cfg.enc_architecture() == "fs_original" or (
            is_legacy and _MODEL_MAPPING[cfg.enc_architecture()].legacy_bgr
            )
        super().__init__(num_identities,
                         input_size,
                         is_rgb=not is_bgr,
                         is_legacy=is_legacy)
        bottleneck_args = (cfg.bottleneck_type(), cfg.bottleneck_size(), cfg.bottleneck_norm())
        bottleneck_in_enc = cfg.bottleneck_in_encoder()
        self.encoder = Encoder(cfg.enc_architecture(),
                               self.input_shape[1],
                               bottleneck_args if bottleneck_in_enc else None,
                               self.is_legacy)

        up_blocks = UpscaleBlocks(
            cfg.output_size(),
            cfg.dec_min_filters(),
            cfg.dec_max_filters(),
            T.cast(T.Literal["full", "cap_max", "cap_min"], cfg.dec_slope_mode()),
            cfg.dec_filter_slope(),
            T.cast(UpsampleT, cfg.dec_upscale_method()),
            cfg.dec_gaussian(),
            T.cast(T.Literal["none", "batch", "group", "instance", "layer", "rms"],
                   cfg.dec_norm()),
            cfg.dec_res_blocks(),
            cfg.dec_skip_last_residual(),
            cfg_loss.learn_mask(),
            self.is_legacy
            )

        self.inter = Inter(self.num_identities,
                           self.encoder.output_shape,
                           cfg.output_size(),
                           cfg.split_fc(),
                           T.cast(T.Literal["none", "full", "half"], cfg.shared_fc()),
                           cfg.fc_min_filters(),
                           cfg.fc_max_filters(),
                           cfg.fc_depth(),
                           cfg.fc_filter_slope(),
                           cfg.fc_dimensions(),
                           cfg.fc_dropout(),
                           T.cast(UpsampleT, cfg.fc_upsampler()),
                           cfg.fc_upsamples(),
                           cfg.fc_upsample_filters(),
                           cfg.dec_upscales_in_fc(),
                           up_blocks,
                           None if bottleneck_in_enc else bottleneck_args,
                           self.is_legacy)
        self.fc_gblock = None
        self.gblock = None
        if cfg.enable_gblock():
            self.fc_gblock = self._get_gblock_inters(
                _get_curve(cfg.fc_gblock_min_nodes(),
                           cfg.fc_gblock_max_nodes(),
                           cfg.fc_gblock_depth(),
                           cfg.fc_gblock_filter_slope()),
                cfg.fc_gblock_dropout(),
                None if bottleneck_in_enc else bottleneck_args
                )
            # TODO make GBlocks module?
            self.gblock = nn.ModuleList(GBlock(self.inter.output_shape[0],
                                               cfg.fc_gblock_max_nodes())
                                        for _ in range(num_identities
                                                       if cfg.split_gblock() else 1))

        self.decoder = Decoder(self.num_identities,
                               cfg.split_decoders(),
                               cfg.dec_upscales_in_fc(),
                               up_blocks,
                               cfg.dec_output_kernel())

    # TODO these 2 properties are hangovers from old system. Revisit when implemented
    @property
    def freeze_layers(self) -> list[str]:
        """ Valid layers to freeze based on configured options """
        return self._select_real_layers(cfg.freeze_layers())

    @property
    def load_layers(self) -> list[str]:
        """ Valid layers to load based on configured options """
        return self._select_real_layers(cfg.load_layers())

    def _get_input_size(self) -> int:
        """ Obtain the input shape for the model.

        Input shape is calculated from the selected Encoder's input size, scaled to the user
        selected Input Scaling, rounded down to the nearest 16 pixels.

        Notes
        -----
        Some models (NasNet) require the input size to be of a certain dimension if loading
        imagenet weights. In these instances resize inputs and raise warning message

        Returns
        -------
        The pixel dimension of the input image
        """
        arch = cfg.enc_architecture()
        enforce_size = _MODEL_MAPPING[arch].enforce_for_weights
        default_size = _MODEL_MAPPING[arch].default_size
        scaling = cfg.enc_scaling() / 100

        min_size = _MODEL_MAPPING[arch].min_size
        size = int(max(min_size, ((default_size * scaling) // 16) * 16))

        if cfg.enc_load_weights() and enforce_size and scaling != 1.0:
            logger.warning("%s requires input size to be %spx when loading imagenet weights. "
                           "Adjusting input size from %spx to %spx",
                           arch, default_size, size, default_size)
            retval = default_size
        else:
            retval = size

        logger.debug("Encoder input size to: %s", retval)
        return retval

    def _validate_encoder_architecture(self) -> None:
        """ Validate that the requested architecture is a valid choice for the running system
        configuration.

        If the selection is not valid, an error is logged and system exits.
        """
        arch = cfg.enc_architecture()
        model = _MODEL_MAPPING.get(arch)
        if not model:
            raise FaceswapError(f"'{arch}' is not a valid choice for encoder architecture. Choose "
                                f"one of {list(_MODEL_MAPPING.keys())}.")
        # TODO keras version tracking removed from here. Delete this when confirmed models are all
        # in valid torch versions

    def _get_gblock_inters(self,
                           feats: list[int],
                           dropout: float,
                           bottleneck_args: tuple[str, int, str] | None) -> nn.Sequential:
        """ obtain the gblock intermediate layers """
        layers = []
        if bottleneck_args:
            assert len(self.encoder.output_shape) == 3
            layers.append(Bottleneck(self.encoder.output_shape, *bottleneck_args, False))
        feats = [self.encoder.output_shape[0] if bottleneck_args is None
                 else layers[-1].output_shape[0]] + feats
        for i, out_feats in enumerate(feats[1:]):
            if dropout > 0.:
                layers.append(nn.Dropout(dropout, inplace=True))
            layers.append(nn.Linear(feats[i], out_feats))
        return nn.Sequential(*layers)

    def forward(self, inputs:  list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...], ...]:
        """ Forward pass through the Phaze-A model

        Parameters
        ---------  -
        inputs: list
            A list of input tensors for the model. This will be of length num_identities with each
            tensor of shape (N, C, H, W)

        Returns
        -------
        The output for each identity training through the model
        """
        encoded = [self.encoder(x) for x in inputs]
        x = self.inter(encoded)
        if self.fc_gblock is not None and self.gblock is not None:
            styles = [self.fc_gblock(enc) for enc in encoded]
            if len(self.gblock) == 1:
                x = [self.gblock[0](content, style) for content, style in zip(x, styles)]
            else:
                x = [gblock(content, style)
                     for gblock, content, style in zip(x, self.gblock, styles)]

        return self.decoder(x)

    # TODO this is a c+p. Needs revisiting when load/freeze weights implemented
    def _select_real_layers(self, layers: list[str]) -> list[str]:
        """ Process the selected freeze or load layers configuration options and replace the
        `keras_encoder` option with the actual keras model name for the configured architecture

        Returns
        -------
        The selected layers for weight freezing
        """
        arch = cfg.enc_architecture()
        # EfficientNetV2 is inconsistent with other model's naming conventions
        keras_name = _MODEL_MAPPING[arch].keras_name.replace("EfficientNetV2", "EfficientNetV2-")
        # CLIPv model is always called 'visual' regardless of weights/format loaded
        keras_name = "visual" if arch.startswith("clipv_") else keras_name

        if "keras_encoder" not in cfg.freeze_layers():
            retval = layers
        elif keras_name:
            retval = [layer.replace("keras_encoder", keras_name) for layer in layers]
            logger.debug("Substituting 'keras_encoder' for '%s'", keras_name)
        else:
            retval = [layer for layer in layers if layer != "keras_encoder"]
            logger.debug("Removing 'keras_encoder' for '%s'", keras_name)

        return retval


__all__ = get_module_objects(__name__)


if __name__ == "__main__":
    from lib.logger import log_setup
    from plugins.train.train_config import load_config
    log_setup("DEBUG", "", "")
    load_config(None)
    INP_SIZE = 64
    cfg.enc_architecture.set(sys.argv[1])
    # cfg.split_fc.set(True)
    # cfg.shared_fc.set("half")
    # cfg.fc_upsampler.set("subpixel")
    mod = PhazeA(2, True)
    print(mod)
    exit()

    inps = [torch.rand((1, 3, INP_SIZE, INP_SIZE)), torch.rand((1, 3, INP_SIZE, INP_SIZE))]
    outs = mod(inps)
    if isinstance(outs[0], tuple):
        print([[x.shape for x in y] for y in outs])
    else:
        print([x.shape for x in outs])

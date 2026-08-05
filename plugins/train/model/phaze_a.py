#!/usr/bin/env python3
""" Phaze-A Model by TorzDF with thanks to BirbFakes and the myriad of testers. """
from __future__ import annotations

import typing as T
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torchvision import models as TVMods

from lib.logger import parse_class_init
from lib.model.layers_legacy import Conv2dLegacy, InstanceNormLegacy, UpSampling2dLegacy
from lib.model.layers import (
    AdaIN, ChannelLayerNorm, ChannelRMSNorm, GaussianNoise, Reshape, ResidualBlock, UpscaleDNY,
    Upscale2xBlock, UpscaleResizeImages, UpscaleSubpixel
)
from lib.model.networks import (  # pylint:disable=unused-import  # noqa:F401
    convnext_xlarge, efficientnet_v2_b0, efficientnet_v2_b1, efficientnet_v2_b2,
    efficientnet_v2_b3, inception_resnet_v2, mobilenet, mobilenet_v3_small, mobilenet_v3_large,
    nasnet_mobile, nasnet_large, resnet50, resnet50_v2, resnet101, resnet101_v2, resnet152,
    resnet152_v2, xception, override_inception3, patch_legacy, vit_b_16, vit_b_32, vit_l_14,
    vit_l_14_336px
)

from lib.utils import FaceswapError, get_module_objects, snake_to_camel_case
from plugins.train.train_config import Loss as cfg_loss

from .base import ModelPlugin
from . import phaze_a_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code,too-many-lines

# TODO summaries instance counts + call per instance counts can be wrong
# TODO warn when some layers could not load weights (imgnet + FS loading) [clipV]


UpsampleT = T.Literal["resize_images", "subpixel", "upscale_dny",
                      "upscale_fast", "upscale_hybrid", "upsample2d"]


@dataclass
class _EncoderInfo:  # pylint:disable=too-many-instance-attributes
    """ Contains model configuration options for various Phaze-A Encoders.

    Parameters
    ----------
    torch_name
        The name of the model in TorchVision. Empty string `""` if the encoder does not exist in
        TorchVision. Names with a preceding tilde ("~") will be loaded locally rather from Torch
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
    legacy_same_pad
        ``True`` if the Keras version of the model implemented TF style asymmetric same-padding.
        Default: ``False``
    legacy_bn_eps
        Value of the Keras version BatchNormalization epsilon. ``None`` if it does not require
        updating. Default: ``None``
    legacy_bn_momentum
        Value of the Keras version BatchNormalization momentum. ``None`` if it does not require
        updating. Default: ``None``
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
    legacy_same_pad: bool = False
    legacy_bn_eps: float | None = None
    legacy_bn_momentum: float | None = None


_CONVNEXT_APPEND = (("classifier", 0), )
_EFF_NET_LEGACY = {"legacy_scaling": (0, 1/255),  # Double scaling bug in legacy
                   "legacy_same_pad": True,
                   "legacy_bn_eps": 1e-3,
                   "legacy_bn_momentum": 0.01}
_EFF_NET_V2_LEGACY = {"legacy_scaling": (-1, 1), "legacy_same_pad": True, "legacy_bn_eps": 1e-3}
_MODEL_MAPPING: dict[str, _EncoderInfo] = {
    "clipv_farl-b-16-16": _EncoderInfo(torch_name="~vit_b_16", kwargs={"weights": "FaRL-B-16-16"}),
    "clipv_farl-b-16-64": _EncoderInfo(torch_name="~vit_b_16", kwargs={"weights": "FaRL-B-16-64"}),
    "clipv_vit-b-16": _EncoderInfo(torch_name="~vit_b_16"),
    "clipv_vit-b-32": _EncoderInfo(torch_name="~vit_b_32"),
    "clipv_vit-l-14": _EncoderInfo(torch_name="~vit_l_14"),
    "clipv_vit-l-14-336px": _EncoderInfo(torch_name="~vit_l_14_336px", default_size=336),
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
    "densenet121": _EncoderInfo(torch_name="densenet121", layer_append=(("nn.relu", None), )),
    "densenet161": _EncoderInfo(torch_name="densenet161", layer_append=(("nn.relu", None), )),
    "densenet169": _EncoderInfo(torch_name="densenet169", layer_append=(("nn.relu", None), )),
    "densenet201": _EncoderInfo(torch_name="densenet201", layer_append=(("nn.relu", None), )),
    "efficientnet_b0": _EncoderInfo(torch_name="efficientnet_b0", **_EFF_NET_LEGACY),
    "efficientnet_b1": _EncoderInfo(
        torch_name="efficientnet_b1", default_size=240, **_EFF_NET_LEGACY),
    "efficientnet_b2": _EncoderInfo(
        torch_name="efficientnet_b2", default_size=260, **_EFF_NET_LEGACY),
    "efficientnet_b3": _EncoderInfo(
        torch_name="efficientnet_b3", default_size=300, **_EFF_NET_LEGACY),
    "efficientnet_b4": _EncoderInfo(
        torch_name="efficientnet_b4", default_size=380, **_EFF_NET_LEGACY),
    "efficientnet_b5": _EncoderInfo(
        torch_name="efficientnet_b5", default_size=456, **_EFF_NET_LEGACY),
    "efficientnet_b6": _EncoderInfo(
        torch_name="efficientnet_b6", default_size=528, **_EFF_NET_LEGACY),
    "efficientnet_b7": _EncoderInfo(
        torch_name="efficientnet_b7", default_size=600, **_EFF_NET_LEGACY),
    "efficientnet_v2_b0": _EncoderInfo(torch_name="~efficientnet_v2_b0", **_EFF_NET_V2_LEGACY),
    "efficientnet_v2_b1": _EncoderInfo(
        torch_name="~efficientnet_v2_b1", default_size=240, **_EFF_NET_V2_LEGACY),
    "efficientnet_v2_b2": _EncoderInfo(
        torch_name="~efficientnet_v2_b2", default_size=260, **_EFF_NET_V2_LEGACY),
    "efficientnet_v2_b3": _EncoderInfo(
        torch_name="~efficientnet_v2_b3", default_size=300, **_EFF_NET_V2_LEGACY),
    "efficientnet_v2_s": _EncoderInfo(
        torch_name="efficientnet_v2_s", default_size=384, **_EFF_NET_V2_LEGACY),
    "efficientnet_v2_m": _EncoderInfo(
        torch_name="efficientnet_v2_m", default_size=480, **_EFF_NET_V2_LEGACY),
    "efficientnet_v2_l": _EncoderInfo(
        torch_name="efficientnet_v2_l", default_size=480, **_EFF_NET_V2_LEGACY),
    "inception_resnet_v2": _EncoderInfo(
        torch_name="~inception_resnet_v2", default_size=299, min_size=75, last_layer="conv2d_7b"),
    "inception_v3": _EncoderInfo(torch_name="inception_v3",
                                 default_size=299,
                                 min_size=75,
                                 kwargs={"aux_logits": False},  # Match keras
                                 last_layer="Mixed_7c",
                                 legacy_scaling=(-1, 1)),
    "mobilenet": _EncoderInfo(torch_name="~mobilenet",
                              last_layer="dw",
                              kwargs={"alpha": cfg.mobilenet_width(),
                                      "depth_multiplier": cfg.mobilenet_depth(),
                                      "dropout": cfg.mobilenet_dropout()}),
    "mobilenet_v2": _EncoderInfo(torch_name="mobilenet_v2",
                                 kwargs={"width_mult": cfg.mobilenet_width()},
                                 legacy_scaling=(-1, 1)),
    "mobilenet_v3_large": _EncoderInfo(torch_name="~mobilenet_v3_large",
                                       legacy_same_pad=True,
                                       kwargs={"width_mult": cfg.mobilenet_width(),
                                               "minimalist": cfg.mobilenet_minimalistic()},
                                       legacy_scaling=(-1, 1)),
    "mobilenet_v3_small": _EncoderInfo(torch_name="~mobilenet_v3_small",
                                       legacy_same_pad=True,
                                       kwargs={"width_mult": cfg.mobilenet_width(),
                                               "minimalist": cfg.mobilenet_minimalistic()},
                                       legacy_scaling=(-1, 1)),
    "nasnet_large": _EncoderInfo(torch_name="~nasnet_large",
                                 kwargs={"include_top": False},
                                 default_size=331,
                                 enforce_for_weights=True),  # TODO check
    "nasnet_mobile": _EncoderInfo(
        torch_name="~nasnet_mobile", kwargs={"include_top": False}, enforce_for_weights=True),  # TODO check
    "resnet50": _EncoderInfo(torch_name="~resnet50", kwargs={"include_top": False}),
    "resnet101": _EncoderInfo(torch_name="~resnet101", kwargs={"include_top": False}),
    "resnet152": _EncoderInfo(torch_name="~resnet152", kwargs={"include_top": False}),
    "resnet50_v1_5": _EncoderInfo(torch_name="resnet50", last_layer="layer4"),
    "resnet101_v1_5": _EncoderInfo(torch_name="resnet101", last_layer="layer4"),
    "resnet152_v1_5": _EncoderInfo(torch_name="resnet152", last_layer="layer4"),
    "resnet50_v2": _EncoderInfo(torch_name="~resnet50_v2", kwargs={"include_top": False}),
    "resnet101_v2": _EncoderInfo(torch_name="~resnet101_v2", kwargs={"include_top": False}),
    "resnet152_v2": _EncoderInfo(torch_name="~resnet152_v2", kwargs={"include_top": False}),
    "vgg16": _EncoderInfo(torch_name="vgg16", legacy_scaling=(0, 255), legacy_bgr=True),
    "vgg19": _EncoderInfo(torch_name="vgg19", legacy_scaling=(0, 255), legacy_bgr=True),
    "xception": _EncoderInfo(torch_name="~xception",
                             min_size=71,
                             default_size=299,
                             last_layer="act4"),
    "fs_original": _EncoderInfo(torch_name="", min_size=32, default_size=1024)
    }


def _calculate_input_size(size: int, scaling: float, version: float) -> int:
    """ Calculate the input shape for the model.

    Parameters
    ----------
    size
        The full size that the model is built for
    scaling
        The amount of scaling that the full input size should be scaled to
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch. Default: 1.0

    Returns
    --------
    The calculated input size, scaled and rounded down to the nearest 16 pixels if scaling is to be
    applied
    """
    if scaling == 1.0 and version >= 1.0:
        return size
    return int(((size * scaling) // 16) * 16)


def _get_curve(start_y: int,
               end_y: int,
               num_points: int,
               scale: float,
               mode: T.Literal["full", "cap_max", "cap_min"] = "full",
               round_to: int = 8) -> list[int]:
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
    round_to
        Round down to the nearest number divisible by the given integer. Default: 8

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
        retval = [int((y // round_to) * round_to) for y in y_axis]
    else:
        y_axis = [start_y]
        scale = 1. - abs(scale)
        for _ in range(num_points - 1):
            current_value = max(end_y, int(((y_axis[-1] * scale) // round_to) * round_to))
            y_axis.append(current_value)
            if current_value == end_y:
                break
        pad = [start_y if mode == "cap_max" else end_y for _ in range(num_points - len(y_axis))]
        retval = pad + y_axis if mode == "cap_max" else y_axis + pad
    logger.debug("Returning curve: %s", retval)
    return retval


def _get_normalization(normalization: str,
                       in_channels: int,
                       version: float) -> nn.Module:
    """ Obtain the selected normalization layer

    Parameters
    ----------
    normalization
        The normalization method to obtain
    in_channels
        The number of features that the normalization will apply to
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch.

    Returns
    -------
    The configured normalization layer
    """
    if normalization == "batch":
        retval = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.01)
    elif normalization == "group":
        if version < 1.0:
            groups = 32  # Was hard-coded for legacy models
        elif in_channels % 16 == 0:
            groups = max(1, in_channels // 16)
        else:  # We always ensure that filters are divisible by 8 in _get_curve
            groups = max(1, in_channels // 8)
        retval = nn.GroupNorm(groups, in_channels, eps=1e-6)
    elif version < 1.0 and normalization == "instance":
        retval = InstanceNormLegacy()
    elif normalization == "instance":
        retval = nn.InstanceNorm2d(in_channels, affine=True)
    elif normalization == "layer":
        retval = ChannelLayerNorm(in_channels, eps=1e-3 if version < 1.0 else 1e-5)
    elif normalization == "rms":
        retval = ChannelRMSNorm(in_channels, eps=1e-8 if version < 1.0 else None)
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
                       version: float = 1.0) -> nn.Module:
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
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch. Default: 1.0

    Returns
    -------
    The selected configured upscale layer
    """
    is_legacy = version < 1.0
    if method == "upsample2d" and is_legacy:
        interpolation = "bilinear" if upsamples > 2 else "nearest"  # Legacy bug
        retval = UpSampling2dLegacy(upsamples, interpolation=interpolation)
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
        retval = UpscaleResizeImages(
            in_channels, out_channels, scale_factor=upsamples, is_legacy=is_legacy
        )
    else:
        raise FaceswapError(
            f"'{method}' is not a valid upscale method. Select from: ['resize_images', "
            "'subpixel', 'upscale_dny', 'upscale_fast', 'upscale_hybrid', 'upsample2d']")
    return retval


class _EncoderFaceswap(nn.Sequential):
    """ A configurable standard Faceswap encoder based off Original model.

    Parameters
    ----------
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch. Default: 1.0
    """
    def __init__(self, version: float) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        depth = cfg.fs_original_depth()
        min_filters = cfg.fs_original_min_filters()
        max_filters = cfg.fs_original_max_filters()
        use_alt = cfg.fs_original_use_alt()

        channels = [3] + [min(max_filters, min_filters * (2 ** i)) for i in range(depth)]
        if use_alt:
            start = channels[0]
            channels = channels[1:] + [channels[-1]]
            self.conv1 = nn.Conv2d(start, channels[0], 1)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)
            for i in range(depth):
                kern = 4 if i == depth - 1 else 3
                pad = 0 if i == depth - 1 else 1
                self.add_module(f"conv{i + 2}", nn.Conv2d(channels[i], channels[i], 3, padding=1))
                self.add_module(f"act{i + 2}", nn.LeakyReLU(0.2, inplace=True))
                self.add_module(
                    f"conv{i + 2}a", nn.Conv2d(channels[i], channels[i + 1], kern, padding=pad)
                    )
                self.add_module(f"act{i + 2}a", nn.LeakyReLU(0.2, inplace=True))
                if i != depth - 1:
                    self.add_module(f"pool{i + 2}", nn.MaxPool2d(2))
        else:
            conv = Conv2dLegacy if version < 1.0 else nn.Conv2d
            padding = "same" if version < 1.0 else 2
            for i in range(depth):
                self.add_module(f"conv{i + 1}",
                                conv(channels[i], channels[i + 1], 5, stride=2, padding=padding))
                self.add_module(f"act{i + 1}",  nn.LeakyReLU(0.1, inplace=True))


class Bottleneck(nn.Sequential):
    """ The bottleneck fully connected layer

    Parameters
    ----------
    input_shape
        The input shape to the bottleneck, excluding batch dimension
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch. Default: 1.0
    """
    def __init__(self, input_shape: tuple[int, int, int], version: float) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        bottleneck = cfg.bottleneck_type()
        size = cfg.bottleneck_size()
        normalization = cfg.bottleneck_norm()

        if normalization and normalization != "none":
            self.norm = _get_normalization(normalization, input_shape[0], version)
        if len(input_shape) > 1 and bottleneck in ("dense", "flatten"):
            self.flat = nn.Flatten()

        output_shape = (input_shape[0], )
        if bottleneck == "dense":
            self.fc = nn.Linear(int(np.prod(input_shape)), size)
            output_shape = (size, )
        elif bottleneck in ("average_pooling", "max_pooling"):
            lyr = nn.MaxPool2d if bottleneck == "max_pooling" else nn.AvgPool2d
            self.pool = lyr(input_shape[1])
            self.flat = nn.Flatten()  # Flatten prior to fc layers

        self.output_shape = output_shape
        """ The output shape from the bottleneck excluding batch dimension """


class _UpscaleGetter:  # pylint:disable=too-many-instance-attributes
    """ Obtain a block of upscalers.

    This class exists outside of the :class:`Decoder` model, as it is possible to place some of
    the upscalers at the end of the Fully Connected Layers, so the upscale chain needs to be able
    to be calculated by both the Fully Connected Layers and by the Decoder if required.

    For this reason, the Upscale Filter list is created early for reference by either the Decoder
    or Fully Connected models
    """
    _reshape_shape: tuple[int, int, int] | None = None
    version: float = 0.0
    input_shape: tuple[int, int, int] | None = None
    _in_channels: list[int] = []
    filters: list[int] = []

    @property
    def out_channels(self) -> int:
        """ The number of filters from the final upscale layer """
        return self.filters[-1]

    @classmethod
    def _calculate_reshape(cls, input_shape: tuple[int, int, int]) -> None:
        """ Calculate whether the input needs reshaping for the chosen model output size and set to
        :attr:`_in_channels` and :attr:`_reshape_shape`

        Parameters
        ----------
        input_shape
            The shape of the Tensor feeding the Upscale Blocks (output from Inter)
        """
        if cls.version >= 1.0:
            # Legacy used to scale filters for mismatched fc_dims and output size. This leads to
            # awkward reshaping between fc_output and decoder input. Now we just set the
            # dimensional space correctly when creating the FC layers
            cls._in_channels.append(input_shape[0])
            logger.debug("[_UpscaleGetter] Set initial in_channel: %s", cls._in_channels)
            return

        old_dim = input_shape[-1]
        new_dim = _scale_dim(cfg.output_size(), old_dim)

        if new_dim == old_dim:
            cls._in_channels.append(input_shape[0])
        else:
            cls._in_channels.append(int(np.prod(input_shape) // new_dim ** 2))
            cls._reshape_shape = (cls._in_channels[0], new_dim, new_dim)
        logger.debug("[_UpscaleGetter] Set initial in_channel: %s and reshape_shape: %s",
                     cls._in_channels, cls._reshape_shape)

    @classmethod
    def _calculate_filters(cls, input_shape: tuple[int, int, int]) -> None:
        """ Generate the filter curve

        Parameters
        ----------
        input_shape
            The shape of the Tensor feeding the Upscale Blocks (output from Inter)
        """
        dim = input_shape[-1] if cls._reshape_shape is None else cls._reshape_shape[-1]
        upscales = int(np.log2(cfg.output_size() / dim))
        cls.filters = _get_curve(cfg.dec_max_filters(),
                                 cfg.dec_min_filters(),
                                 upscales,
                                 cfg.dec_filter_slope(),
                                 mode=T.cast(T.Literal["full", "cap_max", "cap_min"],
                                             cfg.dec_slope_mode()))
        cls._in_channels.extend(cls.filters[:-1])
        logger.debug("[_UpscaleGetter] Set filters: %s (in_channels: %s))",
                     cls.filters, cls._in_channels)

    @classmethod
    def configure(cls,
                  fc_output_shape: tuple[int, int, int],
                  upscales_in_fc: int,
                  shared_fc: bool,
                  version: float) -> None:
        """ Configure the upscale getter

        Parameters
        ----------
        fc_output_shape
            The shape of the Tensor from each fc layer
        upscales_in_fc
            The number of upscales that are being placed within the FC layers
        shared_fc
            ``True`` if the model has a shared fully connected layer
        version
            The plugin version. Versions less than 1.0 means that the model was created in Keras.
            Versions 1.0 and above are created in Torch.
        """
        input_shape = fc_output_shape
        if shared_fc and not upscales_in_fc:  # Input shape is the concatenated inters
            input_shape = (input_shape[0] * 2, *input_shape[1:])

        logger.debug("[_UpscaleGetter] Configuring (fc_output_shape: %s, upscales_in_fc: %s, "
                     "shared_fc: %s, version: %s). Input shape: %s",
                     fc_output_shape, upscales_in_fc, shared_fc, version, input_shape)

        cls.version = version
        cls._calculate_reshape(input_shape)
        cls._calculate_filters(input_shape)

        if not upscales_in_fc or not shared_fc:
            return

        # When upscales are in fully connected layers and there are shared fully connected layers
        # the first in_channel for the decoder upscale requires doubling for the concatenated
        # output.
        # TODO: This unnecessarily blows out filters, but is a legacy hangover. Halving filters
        # for shared upscales in fc would make sense, but as putting upscales in FC for shared FC
        # doesn't make a lot of sense, it's probably not worth handling
        out_idx = upscales_in_fc - 1
        filters = cls.filters[out_idx] * 2
        logger.debug("[_UpscaleGetter] Updating in_channels from %s to %s for %s upscales in fc",
                     cls._in_channels[upscales_in_fc], filters, upscales_in_fc)
        cls._in_channels[upscales_in_fc] = filters

    def _dny_entry(self, layers: dict[str, nn.Module]) -> None:
        """ Entry convolutions for using the upscale_dny method.

        Parameters
        ----------
        layers
            The currently building dict of layers to add the dny entry convs to, if required
        """
        if cfg.dec_upscale_method().lower() != "upscale_dny":
            return
        logger.debug("[_UpscaleGetter] Adding DNY entry layers")
        layers["dny"] = nn.Sequential(
            nn.Conv2d(self._in_channels[0], self.filters[0], 4, padding="same"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.filters[0], self.filters[0], 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

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
        assert self.version

        norm = cfg.dec_norm()
        res_blocks = cfg.dec_res_blocks()
        method = T.cast(UpsampleT, cfg.dec_upscale_method())
        do_act = cfg.dec_upscale_method().lower() in ("subpixel", "upscale_fast", "upscale_hybrid")
        if self.version < 1.0 and cfg.dec_upscale_method().lower() == "resize_images":
            do_act = True  # Legacy bug. Extra leaky-relu

        filters_ = self.filters[index]
        layers: dict[str, nn.Module] = {}

        in_channels = self._in_channels[index]
        if index == 0 and cfg.dec_upscale_method().lower() == "upscale_dny":
            in_channels = self.filters[0]

        logger.debug("[_UpscaleGetter] Creating upscale block %s. in_channels: %s, "
                     "out_channels: %s", index, in_channels, filters_)
        up = _get_upscale_layer(method,
                                in_channels,
                                filters_,
                                upsamples=2,
                                version=self.version)
        layers[method] = up
        if not is_mask and cfg.dec_gaussian():
            layers["noise"] = GaussianNoise(1.0)
        if norm and norm != "none":
            layers["norm"] = _get_normalization(norm, filters_, self.version)

        skip_res = cfg.dec_skip_last_residual() and index == len(self.filters) - 1
        if not is_mask and res_blocks and not skip_res:
            layers["act"] = nn.LeakyReLU(0.2, inplace=True)
            for i in range(res_blocks):
                lbl = str(i + 1) if res_blocks > 1 else ""
                layers[f"res{lbl}"] = ResidualBlock(filters_)
        elif do_act:
            layers["act"] = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(OrderedDict(layers))

    def __call__(self, layer_indices: tuple[int, int] | None = None) -> dict[str, nn.Sequential]:
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
        assert self._in_channels and self.filters, "Input size must be set before calling"

        learn_mask = cfg_loss.learn_mask()
        method = T.cast(UpsampleT, cfg.dec_upscale_method())

        start_idx, end_idx = (0, -1) if layer_indices is None else layer_indices
        end_idx = len(self.filters) if end_idx == -1 else end_idx

        layers: dict[str, nn.Module] = {}
        mask_layers: dict[str, nn.Module] = {}

        if start_idx == 0 and self._reshape_shape is not None:
            layers["reshape"] = Reshape(self._reshape_shape, is_contiguous=False)
            if learn_mask:
                mask_layers["reshape"] = Reshape(self._reshape_shape, is_contiguous=False)

        if start_idx == 0:
            self._dny_entry(layers)
            if learn_mask:
                self._dny_entry(mask_layers)

        for i in range(start_idx, end_idx):
            layers[f"up{i + 1}"] = self._upscale_block(i)
            if learn_mask:
                mask_layers[f"{method}{i + 1}"] = self._upscale_block(i, is_mask=True)
        retval = {"face": nn.Sequential(OrderedDict(layers))}
        if learn_mask:
            retval["mask"] = nn.Sequential(OrderedDict(mask_layers))
        return retval


UPSCALE_GETTER = _UpscaleGetter()
""" Handles returning a sequence of upscalers """


class Encoder(nn.Sequential):
    """ Encoder. Uses one of pre-existing Keras/Faceswap models or custom encoder.

    Parameters
    ----------
    input_size
        The pixel dimension of the encoder input
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch
    """
    def __init__(self, input_size: int, version: float) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        arch = cfg.enc_architecture()
        load_imagenet_weights = cfg.enc_load_weights()
        inc_bottleneck = cfg.bottleneck_in_encoder()

        self._name = snake_to_camel_case(arch)
        mod_info = _MODEL_MAPPING[arch]

        if version < 1.0:  # No need to load weights if we are bringing in a legacy trained model
            load_imagenet_weights = False
        if load_imagenet_weights and not mod_info.torch_name:
            logger.warning("Loading ImageNet weights is not supported for '%s'. "
                           "Weights will be randomly initialized", self._name)
            load_imagenet_weights = False

        self.legacy_scaling = mod_info.legacy_scaling if version < 1.0 else (0, 1)
        setattr(self,
                self._name,
                self._get_encoder(mod_info, load_imagenet_weights, input_size, version))
        out_shape = self._get_output_shape(input_size)
        self.bottleneck = Bottleneck(out_shape, version) if inc_bottleneck else None
        self.output_shape = out_shape if self.bottleneck is None else self.bottleneck.output_shape
        """ The output shape from the encoder excluding batch dimension """

    @property
    def _backbone(self) -> nn.Module:
        """ The encoder backbone """
        return getattr(self, self._name)

    def _get_backbone(self,
                      mod_info: _EncoderInfo,
                      load_weights: bool,
                      input_size: int,
                      version: float
                      ) -> nn.Module:
        """ Load an encoder backbone defined within the Faceswap Repo or in TorchVision """
        if mod_info.torch_name == "inception_v3":
            override_inception3()

        is_local = mod_info.torch_name.startswith("~")
        name = mod_info.torch_name[1:] if is_local else mod_info.torch_name
        module = sys.modules[__name__] if is_local else TVMods
        kwargs = mod_info.kwargs if mod_info.kwargs else {}
        kwargs["weights"] = kwargs.get("weights", "DEFAULT") if load_weights else None
        if self._name.startswith("clipv"):
            kwargs["input_size"] = input_size
        retval: nn.Module = getattr(module, name)(**kwargs)

        if version < 1.0 and (mod_info.legacy_same_pad
                              or mod_info.legacy_bn_eps is not None
                              or mod_info.legacy_bn_momentum is not None):
            patch_legacy(retval,  # Patch the Torch module to be compatible with keras version
                         same_pad=mod_info.legacy_same_pad,
                         bn_eps=mod_info.legacy_bn_eps,
                         bn_momentum=mod_info.legacy_bn_momentum)

        logger.debug("[Encoder] Got backbone: '%s'", retval.__class__.__name__)
        return retval

    def _select_layers(self, backbone: nn.Module, mod_info: _EncoderInfo) -> nn.Module:
        """ Obtain the parts of the backbone that we require for training """
        retval = backbone
        name = retval.__class__.__name__

        if mod_info.last_layer:  # convert all layers to Identity after our final layer
            last_seen = False
            for key, _ in backbone.named_children():
                if last_seen:
                    logger.debug("[Encoder] Setting '%s' layer '%s' to nn.Identity", name, key)
                    setattr(backbone, key, nn.Identity())
                last_seen = last_seen or key == mod_info.last_layer

        if hasattr(backbone, "features"):  # Just take the features part of the model
            logger.debug("[Encoder] Selecting 'features' from '%s' to nn.Identity", name)
            retval = T.cast(nn.Module, backbone.features)

        if mod_info.layer_append:  # Add layers from classifier that are required in the model
            for layer, idx in mod_info.layer_append:
                if layer == "nn.relu":
                    a_name, lyr = "relu", nn.ReLU(inplace=True)
                else:
                    a_name, lyr = f"{layer}{'' if idx is None else idx}", getattr(backbone, layer)
                    lyr = lyr if idx is None else lyr[idx]
                logger.debug("[Encoder] Appending layer for '%s' '%s': '%s'", name, a_name, lyr)
                retval.add_module(a_name, lyr)

        return retval

    def _get_encoder(self,
                     mod_info: _EncoderInfo,
                     load_weights: bool,
                     input_size: int,
                     version: float) -> nn.Module:
        """ Obtain the torch Module for the specified encoder architecture """
        logger.debug("[Encoder] Loading encoder: '%s'", mod_info)
        if mod_info.torch_name:
            backbone = self._get_backbone(mod_info, load_weights, input_size, version)
            retval = self._select_layers(backbone, mod_info)
            return retval

        return _EncoderFaceswap(version)

    def _get_output_shape(self, input_size: int) -> tuple[int, int, int]:
        """ Run a dummy tensor through the model to get the output shape """
        is_train = self._backbone.training
        self._backbone.eval()
        with torch.no_grad():
            input_ = torch.zeros(1, 3, input_size, input_size)
            retval = tuple(self._backbone(input_).shape[1:])
        if is_train:
            self._backbone.train()
        logger.debug("[Encoder] Input size: %s, Output shape: %s", input_size, retval)
        return retval

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint:disable=arguments-renamed
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
        elif self.legacy_scaling == (-1, 1):
            x = x * 2. - 1.  # legacy model expecting inputs from -1 to 1.
        return super().forward(x)


class FullyConnected(nn.Module):
    """ Intermediate Fully Connected layers for Phaze-A Model.

    Parameters
    ----------
    input_shape
        The input shape for the fully connected layers
    dim
        The spatial dimension to reshape the FC outputs to
    feats
        List of number of features for each linear layer
    upsamples
        How many upsamples to apply at the end of the FC layer
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch.
    """
    def __init__(self,  # pylint:disable=too-many-positional-arguments,too-many-arguments,too-many-locals  # noqa:E501
                 input_shape: tuple[int, int, int] | tuple[int],
                 dim: int,
                 feats: list[int],
                 upsample_filters: int,
                 version: float) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        dropout = cfg.fc_dropout()
        upsampler = T.cast(UpsampleT, cfg.fc_upsampler())
        upsamples = cfg.fc_upsamples()
        inc_bottleneck = not cfg.bottleneck_in_encoder()

        if inc_bottleneck:
            assert len(input_shape) == 3
            self.bottleneck = Bottleneck(input_shape, version)
            feats = [self.bottleneck.output_shape[0]] + feats
        else:
            self.bottleneck = None
            feats = [input_shape[0]] + feats

        for i, out_feats in enumerate(feats[1:]):
            lbl = str(i + 1) if len(feats) > 2 else ""
            if dropout > 0.:
                self.add_module(f"drop{lbl}", nn.Dropout(dropout, inplace=True))
            self.add_module(f"fc{lbl}", nn.Linear(feats[i], out_feats))

        dst_shape = (int(feats[-1] / (dim ** 2)), dim, dim)
        self.reshape = Reshape((dst_shape), is_contiguous=True)

        for k, v in self._get_upsamples(
                upsampler, upsamples, dst_shape[0], upsample_filters, version
                ).items():
            self.add_module(k, v)

        out_channels = (dst_shape[0] if not upsamples or upsampler == "upsample2d"
                        else upsample_filters)
        out_dim = dim * 2 ** upsamples
        self.out_shape = (out_channels, out_dim, out_dim)

        self._up_layers: dict[str, nn.Sequential] = {}

    def _get_upsamples(self,
                       upsampler: UpsampleT,
                       upsamples: int,
                       in_channels: int,
                       out_channels: int,
                       version: float) -> dict[str, nn.Module]:
        """ Obtain the upscale layers if requested """
        logger.debug("[FullyConnected] Getting upsamples: %s",
                     {k: v for k, v in locals().items() if k != "self"})
        retval = {}
        if not upsamples:
            if version < 1.0 and upsampler == "upsample2d":  # Bug in keras code
                retval["act"] = nn.LeakyReLU(0.1, inplace=True)
            return retval

        if upsampler == "upsample2d" and upsamples > 1:
            retval[upsampler] = _get_upscale_layer(
                upsampler, in_channels, out_channels, 2 ** upsamples, version
                )
        else:
            for i in range(upsamples):
                lbl = str(i + 1) if upsamples > 1 else ''
                in_c = in_channels if i == 0 else out_channels
                retval[f"{upsampler}{lbl}"] = _get_upscale_layer(
                    upsampler, in_c, out_channels, 2, version
                    )
        if upsampler == "upsample2d":
            retval["act"] = nn.LeakyReLU(0.1, inplace=True)
        logger.debug("[FullyConnected] Upsamples: %s", retval)
        return retval

    @classmethod
    def _fix_legacy_upscale(cls, upscale: nn.Sequential) -> nn.Sequential:
        """ Legacy models have a bug where Residuals are skipped from the final upscale in the
        FC upscales if dec_skip_last_residual is selected """
        if not cfg.dec_skip_last_residual() or not cfg.dec_res_blocks():
            return upscale
        logger.debug("[FullyConnected] Stripping last upscale residual for legacy bug")
        new = OrderedDict()
        for i, (key, val) in enumerate(upscale.named_children()):
            if i != len(upscale) - 1:
                new[key] = val
                continue
            new[key] = nn.Sequential(OrderedDict({k: v for k, v in val.named_children()
                                                  if not k.startswith("res")}))
        del upscale
        return nn.Sequential(new)

    def append_upscales(self, upscales: int, version: float) -> None:
        """ Append decoder upscales to the fully connected layer

        Parameters
        ----------
        upscales
            The number of upscales to append
        version
            The plugin version. Versions less than 1.0 means that the model was created in Keras.
            Versions 1.0 and above are created in Torch.
        """
        for k, v in UPSCALE_GETTER((0, upscales)).items():
            logger.debug("[FullyConnected] Inserting decoder upscale '%s'", k)
            v = self._fix_legacy_upscale(v) if version < 1.0 else v
            self.add_module(k, v)
            self._up_layers[k] = v

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
        for key, module in self.named_children():
            if key in self._up_layers:
                break
            inputs = module(inputs)
        if not self._up_layers:
            return inputs
        out = [up(inputs) for up in self._up_layers.values()]
        return out[0] if len(out) == 1 else tuple(out)


class Inter():
    """ Builds the Fully Connected layers for Phaze-A

    Note
    ----
    This a standard object rather than an nn.Module to work around each of 2 potential (cosmetic)
    issues: 1) The shared inter will be registered twice if this is an nn.module. Whilst the object
    is identical, this leads to duplication in the state_dict which makes weight porting tricky. 2)
    If we build inters for all sides within a single nn.Module, the summary blows out displaying
    each layer for all sides, which is also not what we want

    Parameters
    ----------
    num_identities
        The number of identities the model is being trained on
    input_shape
        The input shape to the FC layers
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch.
    """
    def __init__(self,
                 num_identities: int,
                 input_shape: tuple[int, int, int] | tuple[int],
                 version: float) -> None:
        logger.debug(parse_class_init(locals()))

        self._model_output_size = cfg.output_size()
        self._shared = T.cast(T.Literal["none", "full", "half"], cfg.shared_fc())
        self._split = cfg.split_fc()
        self._dim = cfg.fc_dimensions()
        self._upsamples = cfg.fc_upsamples()

        if num_identities > 2 and self._shared == "half":
            raise FaceswapError("half shared FC layer is not compatible with more than 2"
                                "identities")
        if version >= 1.0 and not self._split and self._shared != "none":
            raise FaceswapError("Shared FC layer is only compatible with split FC layers")

        # Legacy used to just scale filters for mismatched fc_dims and output size. This leads to
        # awkward reshaping between fc_output and decoder input. Now we set the dimensional space
        # correctly staight out of the FC layers
        self._upsample_filters = (
            self._scale_filters(cfg.fc_upsample_filters(),  # For reshape in decoder
                                self._dim * (self._upsamples + 1),
                                cfg.output_size()) if version < 1.0
            else cfg.fc_upsample_filters()  # We reshape here. Use supplied
        )

        self._shared_id = -1  # reference to shared for easier access
        self.modules = self._build_modules(num_identities, input_shape, version)
        """ The inter module/module list """

        self.fc_out_shape = T.cast(FullyConnected, self.modules[0]).out_shape
        """ The output shape from each fully connected layer prior to any decoder upscaling """

        if self._shared != "none":
            self.out_shape = (self.fc_out_shape[0] * 2, *self.fc_out_shape[1:])  # concat filters
            """ The output shape from each inter """
        else:
            self.out_shape = self.fc_out_shape

    @classmethod
    def _scale_filters(cls, original_filters: int, dim: int, output_size: int) -> int:
        """ Scale the filters to be compatible with the model's selected output size.

        Parameters
        ----------
        original_filters
            The original user selected number of filters
        dim
            The original user selected dimensional shape of reshape layer
        output_size
            The final pixel output size from the Phaze-A model

        Returns
        -------
        int
            The number of filters scaled down for output size
        """
        scaled_dim = _scale_dim(output_size, dim)
        if scaled_dim == dim:
            logger.debug("[Inter] filters don't require scaling. Returning: %s", original_filters)
            return original_filters

        flat = dim ** 2 * original_filters
        modifier = dim ** 2 * scaled_dim ** 2
        retval = int((flat // modifier) * modifier)
        retval = int(retval / dim ** 2)
        logger.debug("[Inter] original_filters: %s, scaled_filters: %s", original_filters, retval)
        return retval

    def _get_filters(self, dim: int, upsamples: int, version: float) -> list[int]:
        """ Obtain the filters for each fc layer within each inter """
        # Legacy used to just scale filters by the upscaled output dim. Now we just scale by the
        # original dim
        scale_dim = dim * (upsamples + 1) if version < 1.0 else dim
        min_filters = self._scale_filters(cfg.fc_min_filters(),
                                          scale_dim,
                                          self._model_output_size) * dim ** 2
        max_filters = self._scale_filters(cfg.fc_max_filters(),
                                          scale_dim,
                                          self._model_output_size) * dim ** 2
        retval = _get_curve(min_filters, max_filters, cfg.fc_depth(), cfg.fc_filter_slope())
        logger.debug("[Inter] Got filters: %s", retval)
        return retval

    def _build_modules(self,
                       num_identities: int,
                       input_shape: tuple[int] | tuple[int, int, int],
                       version) -> nn.ModuleList:
        """ Build the inter module list"""
        real_dim = self._dim if version < 1.0 else _scale_dim(self._model_output_size, self._dim)
        fc_args = (input_shape,
                   real_dim,
                   self._get_filters(self._dim, self._upsamples, version),
                   self._upsample_filters,
                   version)

        retval = nn.ModuleList([FullyConnected(*fc_args)])

        if self._split:
            retval.extend([FullyConnected(*fc_args) for _ in range(num_identities - 1)])

        if self._shared == "full":
            retval.append(FullyConnected(*fc_args))
            self._shared_id = len(retval) - 1
        elif self._shared == "half":
            self._shared_id = 0

        return retval

    def configure_upscales(self, upscales: int, version: float) -> None:
        """ Insert decoder upscales into the FC layers + update the inter out_shape if required

        Parameters
        ----------
        upscales
            The number of decoder upscales to place in fc layers
        version
            The plugin version. Versions less than 1.0 means that the model was created in Keras.
            Versions 1.0 and above are created in Torch.
        """
        if not upscales:
            return

        for mod in T.cast(list[FullyConnected], self.modules):
            mod.append_upscales(upscales, version)

        # TODO halve upscale filters in shared
        out_shape = (UPSCALE_GETTER.filters[upscales - 1] * (2 if self._shared != "none" else 1),
                     self.out_shape[1] * (upscales + 1),
                     self.out_shape[2] * (upscales + 1))
        logger.debug("[Inter] Updated output shape from %s to %s for %s upscales",
                     self.out_shape, out_shape, upscales)
        self.out_shape = out_shape

    def __call__(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """ Forward pass through the intermediate layers

        Parameters
        ----------
        inputs
            The input tensors for each side to the Intermediate layers

        Returns
        -------
        The output tensor from each side's Intermediate layer
        """
        if self._split:
            x = [inter(i) for inter, i in zip(self.modules[:len(inputs)], inputs)]
        else:
            x = [self.modules[0](i) for i in inputs]

        if self._shared_id < 0:
            return x

        return [torch.concat([self.modules[self._shared_id](i), y], dim=1)
                for i, y in zip(inputs, x)]


class InterGBlock(nn.Sequential):
    """ Intermediate block that feeds the Style part of G-Block

    Parameters
    ----------
    input_shape
        The input shape to the linear layers feeding the G-Block
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch.
    """
    def __init__(self, input_shape: tuple[int, int, int] | tuple[int], version: float) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        depth = cfg.fc_gblock_depth()
        min_nodes = cfg.fc_gblock_min_nodes()
        max_nodes = cfg.fc_gblock_max_nodes()
        slope = cfg.fc_gblock_filter_slope()
        dropout = cfg.fc_dropout()
        inc_bottleneck = not cfg.bottleneck_in_encoder()

        in_channels = input_shape[0]
        if inc_bottleneck:
            assert len(input_shape) == 3
            bottleneck = Bottleneck(input_shape, version)
            self.add_module("bottleneck", bottleneck)
            in_channels = bottleneck.output_shape[0]

        fc_feats = _get_curve(min_nodes, max_nodes, depth, slope)
        for i, feats in enumerate(fc_feats):
            if dropout > 0.:
                self.add_module(f"drop{i}", nn.Dropout(dropout, inplace=True))
            self.add_module(f"fc{i}", nn.Linear(in_channels if i == 0 else fc_feats[i - 1], feats))

        self.in_channels = in_channels
        """ The number of output channels from the encoder + bottleneck """


class GBlock(nn.Module):
    """ G-Block model, borrowing from Adain StyleGAN.

    Parameters
    ----------
    style_channels
        The number of channels feeding the style part of G-Block
    content_channels
        The number of channels feeding the content part of G-Block
    style_recursions
        The number of recursions for the style Linear layers. Default: 3
    style_nodes
        The number of channels in each style Linear layer. Default: 512
    g_block_recursions
        The number of recursions to perform within the G-Block. Default: 2
    """
    def __init__(self,
                 style_channels: int,
                 content_channels: int,
                 style_recursions: int = 3,
                 style_nodes: int = 512,
                 g_block_recursions: int = 2) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._gblock_recursions = g_block_recursions
        style = []
        for i in range(style_recursions):
            style.append(nn.Linear(style_channels if i == 0 else style_nodes, style_nodes))
            if i != style_recursions - 1:
                style.append(nn.LeakyReLU(0.1, inplace=True))
        self.style = nn.Sequential(*style)
        self.content = nn.Sequential(nn.Conv2d(content_channels, content_channels, 3, padding=1),
                                     GaussianNoise(1.0))
        self.gblock = nn.ModuleList()
        for i in range(self._gblock_recursions):
            gblock = nn.ModuleDict()
            gblock["style"] = nn.ModuleList((
                nn.Sequential(nn.Linear(style_channels, content_channels),
                              Reshape((content_channels, 1, 1), is_contiguous=True))
                for _ in range(2)))
            gblock["noise"] = nn.Sequential(
                GaussianNoise(1.0),
                nn.Conv2d(content_channels, content_channels, 1)
                )
            if i == self._gblock_recursions - 1:
                gblock["conv"] = nn.Conv2d(content_channels, content_channels, 3, padding=1)
            gblock["norm"] = AdaIN(dim=1)
            gblock["act"] = nn.LeakyReLU(0.2, inplace=True)
            self.gblock.append(gblock)

    def forward(self, style: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the G-Block

        Parameters
        ----------
        style
            The output from inter_gblock
        content
            The outputs from the Phaze-A Intermediate layer

        Returns
        -------
        The output tensor from the G-Block
        """
        style = self.style(style)
        x = self.content(content)
        for idx, gblock in enumerate(T.cast(list[nn.ModuleDict], self.gblock)):
            styles = tuple(s(style) for s in T.cast(nn.ModuleList, gblock["style"]))
            noise = gblock["noise"](x)
            if idx == self._gblock_recursions - 1:
                x = gblock["conv"](x)
            norm = gblock["norm"](x, styles)
            x = gblock["act"](norm + noise)
        return x


class Decoder(nn.ModuleDict):
    """ The Decoder(s) for Phaze-A

    Parameters
    ----------
    upscales_in_fc
        The number of upscales placed into the fully connected layers
    output_kernel
        The size of the kernel for the final conv layer
    """
    def __init__(self, upscales_in_fc: int, output_kernel: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = False
        upscale_blocks = UPSCALE_GETTER((upscales_in_fc, -1) if upscales_in_fc else None)
        for key, up in upscale_blocks.items():
            out_channels = 3 if key == "face" else 1
            if key == "mask":
                self.learn_mask = True
            up.add_module("conv", nn.Conv2d(UPSCALE_GETTER.out_channels,
                                            out_channels,
                                            output_kernel,
                                            padding="same"))
            up.add_module("act", nn.Sigmoid())
            self.add_module(key, up)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Call the Phaze-A Decoder(s)

        Parameters
        ----------
        inputs
            The input tensors to the Phaze-A Decoder(s)

        Returns
        -------
        The output tensors from the Phaze-A Decoder(s)
        """
        return tuple(module(inputs) for module in self.values())


class PhazeA(ModelPlugin):
    """ Phaze-A Faceswap Model.

    An highly adaptable and configurable model by torzDF

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch. Default: 1.0
    """
    def __init__(self, num_identities: int = 2, version=1.0) -> None:
        if cfg.output_size() % 16 != 0:
            raise FaceswapError("Phaze-A output shape must be a multiple of 16")

        self._validate_encoder_architecture()
        self._split_fc = cfg.split_fc()
        self._shared_fc = cfg.shared_fc() != "none"
        self._split_gblock = cfg.split_gblock()
        self._split_decoders = cfg.split_decoders()

        input_size = self._get_input_size(version)
        is_bgr = cfg.enc_architecture() == "fs_original" or (
            version < 1.0 and _MODEL_MAPPING[cfg.enc_architecture()].legacy_bgr
            )
        super().__init__(num_identities,
                         input_size,
                         version=version,
                         is_rgb=not is_bgr)

        self.encoder = Encoder(self.input_shape[1], version)

        self._inter = Inter(num_identities, self.encoder.output_shape, version)
        self.inter = self._inter.modules  # Elevate to parent
        upscales_in_fc = cfg.dec_upscales_in_fc()
        UPSCALE_GETTER.configure(self._inter.fc_out_shape,
                                 upscales_in_fc,
                                 self._shared_fc,
                                 version)
        self._inter.configure_upscales(upscales_in_fc, version)

        self.inter_gblock, self.gblock = self._build_gblock()
        self.decoder = self._build_decoder()

    # TODO these 2 properties are hangovers from old system. Revisit when implemented
    @property
    def freeze_layers(self) -> list[str]:
        """ Valid layers to freeze based on configured options """
        return self._select_real_layers(cfg.freeze_layers())

    @property
    def load_layers(self) -> list[str]:
        """ Valid layers to load based on configured options """
        return self._select_real_layers(cfg.load_layers())

    def _get_input_size(self, version: float) -> int:
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
        size = int(max(min_size, _calculate_input_size(default_size, scaling, version)))
        if cfg.enc_load_weights() and enforce_size and scaling != 1.0:
            logger.warning("%s requires input size to be %spx when loading imagenet weights. "
                           "Adjusting input size from %spx to %spx",
                           arch, default_size, size, default_size)
            retval = default_size
        else:
            retval = size
        logger.debug("Encoder input size: %s", retval)
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

    def _build_gblock(self) -> tuple[InterGBlock, GBlock | nn.ModuleList] | tuple[None, None]:
        """ Build the gblock """
        if not cfg.enable_gblock():
            return None, None

        inter = InterGBlock(self.encoder.output_shape, self.version)
        content_channels = self._inter.out_shape[0]
        if not self._split_gblock:
            gblock = GBlock(cfg.fc_gblock_max_nodes(), content_channels)
        else:
            gblock = nn.ModuleList([GBlock(cfg.fc_gblock_max_nodes(), content_channels)
                                    for _ in range(self.num_identities)])
        return inter, gblock

    def _build_decoder(self) -> Decoder | nn.ModuleList:
        """ Build the Phaze-A decoder """
        upscales_in_fc = cfg.dec_upscales_in_fc()
        out_kernel = cfg.dec_output_kernel()
        if self._split_decoders:
            return nn.ModuleList([Decoder(upscales_in_fc, out_kernel)
                                  for _ in range(self.num_identities)])
        return Decoder(upscales_in_fc, out_kernel)

    def forward(self, inputs:  list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
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
        x = self._inter(encoded)

        if self.inter_gblock is not None and self.gblock is not None:
            styles = [self.inter_gblock(enc) for enc in encoded]
            if self._split_gblock:
                x = [g(s, c) for g, s, c in zip(T.cast(nn.ModuleList, self.gblock), styles, x)]
            else:
                x = [self.gblock(s, c) for s, c in zip(styles, x)]

        if self._split_decoders:
            out = [dec(y) for dec, y in zip(T.cast(nn.ModuleList, self.decoder), x)]
        else:
            out = [self.decoder(y) for y in x]
        return tuple(out)

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

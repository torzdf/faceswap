#! /usr/env/bin/python3
""" Reconfiguration of certain TorchVision models for migrating from Keras Application and making
compatible with faceswap """
from __future__ import annotations

from collections import OrderedDict
import logging
import typing as T

import torch
from torch import nn
from torchvision import models as tv_mods

from lib.utils import get_module_objects
from lib.model.layers_legacy import Conv2dLegacy
from lib.model.weights import GetWeights

logger = logging.getLogger(__name__)


def patch_legacy(model: nn.Module,
                 same_pad: bool = False,
                 bn_eps: float | None = None,
                 bn_momentum: float | None = None,
                 parent: nn.Module | None = None) -> None:
    """ Update a Torch model for Keras style 'same' padding and batch_norm params for legacy
    imported models

    Parameters
    ----------
    model
        The EfficientNet model to update
    same_pad
        ``True`` to update convolutions with stride > 1 and padding > 0 to TF style asymmetric
        padding. Default: ``False``
    bn_eps
        The epsilon to update BatchNorm2d layers to. ``None`` for no update
    bn_momentum
        The momentum to update BatchNorm2d layers to. ``None`` for no update
    parent
        The parent module that owns the convolution to be updated. Default: ``None`` (First
        recursion)
    """
    for name, mod in model.named_children():
        # pylint:disable=too-many-boolean-expressions,unidiomatic-typecheck
        if same_pad and (type(mod) is nn.Conv2d and any(x > 1 for x in mod.stride)
                         and (mod.padding == "same" or (isinstance(mod.padding, tuple)
                                                        and any(x > 0 for x in mod.padding)))):
            logger.debug("Updating '%s.%s' for legacy 'same' padding: %s",
                         parent.__class__.__name__, name, mod)
            new_conv = Conv2dLegacy(mod.in_channels,
                                    mod.out_channels,
                                    T.cast(tuple[int, int], mod.kernel_size),
                                    stride=T.cast(tuple[int, int], mod.stride),
                                    padding="same",
                                    dilation=T.cast(tuple[int, int], mod.dilation),
                                    groups=mod.groups,
                                    bias=mod.bias is not None)
            new_conv.load_state_dict(mod.state_dict())
            setattr(model, name, new_conv)
            del mod
        elif isinstance(mod, nn.BatchNorm2d) and (
                (bn_eps is not None and mod.eps != bn_eps) or
                (bn_momentum is not None and mod.momentum != bn_momentum)
                ):
            logger.debug("Updating '%s.%s' for legacy 'eps'(%s) and 'momentum'(%s): %s",
                         parent.__class__.__name__, name, mod.eps if bn_eps is None else bn_eps,
                         mod.momentum if bn_momentum is None else bn_momentum, mod)
            mod.eps = 1e-3
            mod.momentum = 0.1
        else:
            patch_legacy(mod,
                         same_pad=same_pad,
                         bn_eps=bn_eps,
                         bn_momentum=bn_momentum,
                         parent=model)


def load_imagenet_weights(model: nn.Module,
                          weights: T.Literal["DEFAULT"] | None,
                          file_name: str,
                          skip: list[str] | None = None) -> None:
    """ Load ImageNet weights into the model if specified. The weights must have the key "DEFAULT"
    to load ImageNet weights or ``None`` to not load any weights.

    Parameters
    ----------
    model
        The model to load weights into.
    weights
        The weights to load. If "DEFAULT", load the default ImageNet weights.
        If None, do not load any weights.
    file_name
        The name of the weights file.
    skip

    """
    if weights is None:
        return

    if weights != "DEFAULT":
        logger.warning("Invalid weights type: '%s'. Falling back to 'DEFAULT'", weights)
        weights = "DEFAULT"

    weights_file = GetWeights(file_name).model_path
    assert isinstance(weights_file, str)

    state_dict: OrderedDict[str, torch.Tensor] = torch.load(weights_file,
                                                            map_location="cpu")

    strict = True
    if skip is not None and any(x.startswith(y) for y in skip for x in state_dict):
        state_dict = OrderedDict((k, v) for k, v in state_dict.items()
                                 if not any(k.startswith(y) for y in skip))
        strict = False

    model.load_state_dict(state_dict, strict=strict)


def convnext_xlarge(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                    ) -> tv_mods.convnext.ConvNeXt:
    """ ConvNext X-Large settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    block_setting = [tv_mods.convnext.CNBlockConfig(256, 512, 3),
                     tv_mods.convnext.CNBlockConfig(512, 1024, 3),
                     tv_mods.convnext.CNBlockConfig(1024, 2048, 27),
                     tv_mods.convnext.CNBlockConfig(2048, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    retval = tv_mods.convnext.ConvNeXt(block_setting, stochastic_depth_prob, **kwargs)

    load_imagenet_weights(retval, weights, "convnext_xlarge_imagenet.pth")
    return retval


def efficientnet_v2_b0(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> tv_mods.efficientnet.EfficientNet:
    """ EfficientNetV2_b0 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    inverted_residual_setting = [tv_mods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 1),
                                 tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 2),
                                 tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 48, 2),
                                 tv_mods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3),
                                 tv_mods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5),
                                 tv_mods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8)]
    dropout = kwargs.pop("dropout", 0.2)
    retval = tv_mods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1280, **kwargs
    )
    load_imagenet_weights(retval, weights, "efficientnet_v2_b0_imagenet.pth")
    return retval


def efficientnet_v2_b1(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> tv_mods.efficientnet.EfficientNet:
    """ EfficientNetV2_b1 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    inverted_residual_setting = [
        tv_mods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 2),
        tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 3),
        tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 48, 3),
        tv_mods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3, depth_mult=1.1),
        tv_mods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5, depth_mult=1.1),
        tv_mods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8, depth_mult=1.1)
    ]
    dropout = kwargs.pop("dropout", 0.2)
    retval = tv_mods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1280, **kwargs
    )
    load_imagenet_weights(retval, weights, "efficientnet_v2_b1_imagenet.pth")
    return retval


def efficientnet_v2_b2(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> tv_mods.efficientnet.EfficientNet:
    """ EfficientNetV2_b2 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    inverted_residual_setting = [
        tv_mods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 2),
        tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 3),
        tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 56, 3),
        tv_mods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3, width_mult=1.1, depth_mult=1.2),
        tv_mods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5, width_mult=1.1, depth_mult=1.2),
        tv_mods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8, width_mult=1.1, depth_mult=1.2)
    ]
    dropout = kwargs.pop("dropout", 0.2)
    retval = tv_mods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1408, **kwargs
    )
    load_imagenet_weights(retval, weights, "efficientnet_v2_b2_imagenet.pth")
    return retval


def efficientnet_v2_b3(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> tv_mods.efficientnet.EfficientNet:
    """ EfficientNetV2_b3 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    inverted_residual_setting = [
        tv_mods.efficientnet.FusedMBConvConfig(1, 3, 1, 40, 16, 2),
        tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 40, 3),
        tv_mods.efficientnet.FusedMBConvConfig(4, 3, 2, 40, 56, 3),
        tv_mods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3, width_mult=1.2, depth_mult=1.4),
        tv_mods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5, width_mult=1.2, depth_mult=1.4),
        tv_mods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8, width_mult=1.2, depth_mult=1.4)
    ]
    dropout = kwargs.pop("dropout", 0.2)
    retval = tv_mods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1536, **kwargs
        )
    load_imagenet_weights(retval, weights, "efficientnet_v2_b3_imagenet.pth")
    return retval


__all__ = get_module_objects(__name__)

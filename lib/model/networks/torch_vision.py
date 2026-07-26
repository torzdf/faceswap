#! /usr/env/bin/python3
""" Reconfiguration of certain TorchVision models for migrating from Keras Application and making
compatible with faceswap """
from __future__ import annotations

import logging
import typing as T

from torch import nn
from torchvision import models as TVMods

from lib.utils import get_module_objects
from lib.model.layers_legacy import Conv2dLegacy

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


def convnext_xlarge(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                    ) -> TVMods.convnext.ConvNeXt:
    """ ConvNext X-Large settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
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
    """ EfficientNetV2_b0 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
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


def efficientnet_v2_b1(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> TVMods.efficientnet.EfficientNet:
    """ EfficientNetV2_b1 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    inverted_residual_setting = [
        TVMods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 2),
        TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 3),
        TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 48, 3),
        TVMods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3, depth_mult=1.1),
        TVMods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5, depth_mult=1.1),
        TVMods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8, depth_mult=1.1)
    ]
    dropout = kwargs.pop("dropout", 0.2)
    retval = TVMods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1280, **kwargs
    )
    # TODO port weights and load here
    return retval


def efficientnet_v2_b2(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> TVMods.efficientnet.EfficientNet:
    """ EfficientNetV2_b2 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    inverted_residual_setting = [
        TVMods.efficientnet.FusedMBConvConfig(1, 3, 1, 32, 16, 2),
        TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 32, 3),
        TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 32, 56, 3),
        TVMods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3, width_mult=1.1, depth_mult=1.2),
        TVMods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5, width_mult=1.1, depth_mult=1.2),
        TVMods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8, width_mult=1.1, depth_mult=1.2)
    ]
    dropout = kwargs.pop("dropout", 0.2)
    retval = TVMods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1408, **kwargs
    )
    # TODO port weights and load here
    return retval


def efficientnet_v2_b3(weights: T.Literal["DEFAULT"] | None = None, **kwargs: T.Any
                       ) -> TVMods.efficientnet.EfficientNet:
    """ EfficientNetV2_b3 settings from Keras that does not exit in Torch

    Parameters
    ----------
    weights
        "DEFAULT" to load imagenet trained weights
    """
    inverted_residual_setting = [
        TVMods.efficientnet.FusedMBConvConfig(1, 3, 1, 40, 16, 2),
        TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 16, 40, 3),
        TVMods.efficientnet.FusedMBConvConfig(4, 3, 2, 40, 56, 3),
        TVMods.efficientnet.MBConvConfig(4, 3, 2, 48, 96, 3, width_mult=1.2, depth_mult=1.4),
        TVMods.efficientnet.MBConvConfig(6, 3, 1, 96, 112, 5, width_mult=1.2, depth_mult=1.4),
        TVMods.efficientnet.MBConvConfig(6, 3, 2, 112, 192, 8, width_mult=1.2, depth_mult=1.4)
    ]
    dropout = kwargs.pop("dropout", 0.2)
    retval = TVMods.efficientnet.EfficientNet(
        inverted_residual_setting, dropout, last_channel=1536, **kwargs
        )
    # TODO port weights and load here
    return retval


__all__ = get_module_objects(__name__)

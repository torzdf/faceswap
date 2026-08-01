#!/usr/bin/env python3
""" Original Model
Based on the original https://www.reddit.com/r/deepfakes/ code sample + contributions.

This model is heavily documented as it acts as a template that other model plugins can be developed
from.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers import Reshape, UpscaleSubpixel
from lib.model.layers_legacy import Conv2dLegacy
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss

from .base import ModelPlugin
from . import original_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Encoder(nn.Sequential):  # pylint:disable=too-many-instance-attributes
    """ The original Encoder

    Parameters
    ----------
    low_mem
        ``True`` for low memory version
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self, low_mem: bool, is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        conv = Conv2dLegacy if is_legacy else nn.Conv2d
        padding = "same" if is_legacy else 2

        self.conv1 = conv(3, 128, 5, stride=2, padding=padding)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv(128, 256, 5, stride=2, padding=padding)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = conv(256, 512, 5, stride=2, padding=padding)
        self.act3 = nn.LeakyReLU(0.1, inplace=True)
        if not low_mem:
            self.conv4 = conv(512, 1024, 5, stride=2, padding=padding)
            self.act4 = nn.LeakyReLU(0.1, inplace=True)

        feats = 512 if low_mem else 1024
        in_dim = 8 if low_mem else 4
        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(feats * in_dim * in_dim, feats)
        self.dense2 = nn.Linear(feats, 1024 * 4 * 4)
        self.reshape = Reshape((1024, 4, 4), is_contiguous=True)

        self.up = UpscaleSubpixel(1024, 512)


class Decoder(nn.Module):
    """ The original Faceswap Decoder Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self, learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask
        self.up1 = UpscaleSubpixel(512, 256)
        self.up2 = UpscaleSubpixel(256, 128)
        self.up3 = UpscaleSubpixel(128, 64)
        self.conv = nn.Conv2d(64, 3, 5, stride=1, padding=2)

        if learn_mask:
            self.mask_up1 = UpscaleSubpixel(512, 256)
            self.mask_up2 = UpscaleSubpixel(256, 128)
            self.mask_up3 = UpscaleSubpixel(128, 64)
            self.mask_conv = nn.Conv2d(64, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the Faceswap decoder

        Parameters
        ----------
        inputs
            The input to the Decoder

        Returns
        -------
        outputs
            The image output and optionally mask from the decoder
        """
        x = self.up1(inputs)
        x = self.up2(x)
        x = self.up3(x)
        x = torch.sigmoid(self.conv(x))

        if not self.learn_mask:
            return (x, )

        mask = self.mask_up1(inputs)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class Original(ModelPlugin):
    """ Original Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, num_identities: int = 2, is_legacy: bool = False) -> None:
        super().__init__(num_identities, input_size=64, is_legacy=is_legacy)
        self.encoder = Encoder(cfg.lowmem(), self.is_legacy)
        self.decoders = nn.ModuleList(Decoder(cfg_loss.learn_mask())
                                      for _ in range(num_identities))

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[tuple[torch.Tensor, ...]]:
        """ Forward pass through the original model

        Parameters
        ----------
        inputs: list
            A list of input tensors for the model. This will be of length num_identities with each
            tensor of shape (N, C, H, W)

        Returns
        -------
        The output for each identity training through the model
        """
        encoded = [self.encoder(x) for x in inputs]
        decoded = tuple(dec(x) for dec, x in zip(self.decoders, encoded))
        return decoded


__all__ = get_module_objects(__name__)

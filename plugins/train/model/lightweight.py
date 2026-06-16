#!/usr/bin/env python3
""" Lightweight Model by torzdf
    An extremely limited model for training on low-end graphics cards
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contributions """

from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.nn_blocks import ConvBlockLegacy, UpscaleSubpixel
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss
from .base import ModelPlugin

# pylint:disable=duplicate-code

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """The lightweight Encoder"""
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.conv1 = ConvBlockLegacy(3, 128, 5, stride=2, padding="same")
        self.conv2 = ConvBlockLegacy(128, 256, 5, stride=2, padding="same")
        self.conv3 = ConvBlockLegacy(256, 512, 5, stride=2, padding="same")
        self.dense1 = nn.Linear(512 * 8 * 8, 512)
        self.dense2 = nn.Linear(512, 512 * 4 * 4)
        self.up = UpscaleSubpixel(512, 256)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Lightweight encoder

        Parameters
        ----------
        inputs
            The input to the encoder

        Returns
        -------
        The output from the encoder
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x.flatten(start_dim=1))
        x = T.cast(torch.Tensor, self.dense2(x))
        x = x.view(x.shape[0], 512, 4, 4)
        return self.up(x)


class Decoder(nn.Module):
    """The Lightweight Faceswap Decoder Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self, learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.up1 = UpscaleSubpixel(256, 512)
        self.up2 = UpscaleSubpixel(512, 256)
        self.up3 = UpscaleSubpixel(256, 128)
        self.conv = nn.Conv2d(128, 3, 5, stride=1, padding=2)

        self.mask_up1 = self.mask_up2 = self.mask_up3 = None
        if learn_mask:
            self.mask_up1 = UpscaleSubpixel(256, 512)
            self.mask_up2 = UpscaleSubpixel(512, 256)
            self.mask_up3 = UpscaleSubpixel(256, 128)
            self.mask_conv = nn.Conv2d(128, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass through the Faceswap decoder

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

        if self.up_mask1 is None:
            return (x, )

        assert (self.mask_up1 is not None and
                self.mask_up2 is not None and
                self.mask_up3 is not None)
        mask = self.mask_up1(inputs)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class Lightweight(ModelPlugin):
    """ Lightweight Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    """
    def __init__(self, num_identities: int = 2) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(num_identities, input_size=64)
        self.encoder = Encoder()
        self.decoders = nn.ModuleList(Decoder(cfg_loss.learn_mask())
                                      for _ in range(num_identities))

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...]]:
        """Forward pass through the original model

        Parameters
        ----------
        inputs: list
            A list of input tensors for the model. This will be of length num_identities with each
            tensor of shape (N, C, H, W)

        Returns
        -------
        The output for each identity training through the model
        """
        return tuple(dec(self.encoder(x)) for x, dec in zip(inputs, self.decoders))


__all__ = get_module_objects(__name__)

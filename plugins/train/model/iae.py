#!/usr/bin/env python3
""" Improved autoencoder for faceswap """
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers_legacy import ConvBlockLegacy
from lib.model.nn_blocks import UpscaleSubpixel
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss
from .base import ModelPlugin


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Encoder(nn.Module):
    """ The IAE Encoder

    Parameters
    ----------
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self, is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        if is_legacy:
            self.conv1 = ConvBlockLegacy(3, 128, 5, stride=2, padding="same")
            self.conv2 = ConvBlockLegacy(128, 256, 5, stride=2, padding="same")
            self.conv3 = ConvBlockLegacy(256, 512, 5, stride=2, padding="same")
            self.conv4 = ConvBlockLegacy(512, 1024, 5, stride=2, padding="same")
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 128, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(512, 1024, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the IAE encoder

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
        x = self.conv4(x)
        return self.flatten(x)


class Intermediate(nn.Module):
    """ The IAE Intermediate Network """
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.dense1 = nn.Linear(1024 * 4 * 4, 1024)
        self.dense2 = nn.Linear(1024, 512 * 4 * 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the IAE Intermediate layer

        Parameters
        ----------
        inputs
            The input to the intermediate layer

        Returns
        -------
        The output from the intermediate layer
        """
        x = self.dense1(inputs)
        x = T.cast(torch.Tensor, self.dense2(x))
        return x.view(x.shape[0], 512, 4, 4)


class Decoder(nn.Module):
    """ The IAE Faceswap Decoder Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self, learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.up1 = UpscaleSubpixel(1024, 512)
        self.up2 = UpscaleSubpixel(512, 256)
        self.up3 = UpscaleSubpixel(256, 128)
        self.up4 = UpscaleSubpixel(128, 64)
        self.conv = nn.Conv2d(64, 3, 5, stride=1, padding=2)

        self.mask_up1 = self.mask_up2 = self.mask_up3 = None
        if learn_mask:
            self.mask_up1 = UpscaleSubpixel(1024, 512)
            self.mask_up2 = UpscaleSubpixel(512, 256)
            self.mask_up3 = UpscaleSubpixel(256, 128)
            self.mask_up4 = UpscaleSubpixel(128, 64)
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
        x = self.up4(x)
        x = torch.sigmoid(self.conv(x))

        if self.mask_up1 is None:
            return (x, )

        assert (self.mask_up1 is not None and
                self.mask_up2 is not None and
                self.mask_up3 is not None and
                self.mask_up4 is not None)
        mask = self.mask_up1(inputs)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = self.mask_up4(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class IAE(ModelPlugin):
    """ IAE Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, num_identities: int = 2, is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(num_identities, input_size=64, is_legacy=is_legacy)
        self.encoder = Encoder(self.is_legacy)
        self.inter_both = Intermediate()
        self.inter_side = nn.ModuleList(Intermediate() for _ in range(num_identities))
        self.decoder = Decoder(cfg_loss.learn_mask())

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...]]:
        """ Forward pass through the IAE model

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
        inters = [torch.concat([int(enc), self.inter_both(enc)], dim=1)
                  for enc, int in zip(encoded, self.inter_side)]
        decoded = tuple(self.decoder(x) for x in inters)
        return decoded


__all__ = get_module_objects(__name__)

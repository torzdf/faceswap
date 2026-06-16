#!/usr/bin/env python3
""" Original - VillainGuy model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contributions
    Adapted from a model by VillainGuy (https://github.com/VillainGuy) """
from __future__ import annotations

import logging

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers import SeparableConv2d
from lib.model.nn_blocks import ConvBlockLegacy, ResidualBlock, UpscaleSubpixel
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss
from .base import ModelPlugin
from . import original_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Encoder(nn.Module):  # pylint:disable=too-many-instance-attributes
    """The Villain Encoder

    Parameters
    ----------
    low_mem
        ``True`` for low memory version
    """
    def __init__(self, low_mem: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.feats = 512 if low_mem else 1024

        self.down1 = ConvBlockLegacy(3, 128, 5, stride=2, padding="same", leaky_slope=-1.)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.res = nn.Sequential(*(ResidualBlock(128, 128) for _ in range(8 if low_mem else 16)))
        self.leaky2 = nn.LeakyReLU(0.1)
        self.down2 = nn.Sequential(ConvBlockLegacy(128, 128, 5, stride=2, padding="same"),
                                   nn.PixelShuffle(2))
        self.down3 = nn.Sequential(ConvBlockLegacy(32, 128, 5, stride=2, padding="same"),
                                   nn.PixelShuffle(2))
        self.down4 = nn.Sequential(
            ConvBlockLegacy(32, 128, 5, stride=2, padding="same"),
            SeparableConv2d(128, 256, 5, stride=2, padding=2, is_legacy=True)
            )

        self.down5 = ConvBlockLegacy(256, 512, 5, stride=2, padding="same")
        if not low_mem:
            self.down5 = nn.Sequential(
                self.down5,
                SeparableConv2d(512, 1024, 5, stride=2, padding=2, is_legacy=True)
            )

        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(self.feats * 4 * 4, self.feats)
        self.dense2 = nn.Linear(self.feats, 1024 * 8 * 8)
        self.up = UpscaleSubpixel(1024, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Villain encoder

        Parameters
        ----------
        inputs
            The input to the encoder

        Returns
        -------
        The output from the encoder
        """
        x = self.down1(inputs)
        tmp_x = x

        x = self.leaky1(x)
        x = self.res(x)

        tmp_x = self.leaky2(tmp_x)
        x = x + tmp_x

        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x: torch.Tensor = self.dense2(x)
        x = x.view(x.shape[0], 1024, 8, 8)
        return self.up(x)


class Decoder(nn.Module):
    """The Villain Faceswap Decoder Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self, learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask
        self.up1 = nn.Sequential(UpscaleSubpixel(512, 512, leaky_slope=0.2),
                                 ResidualBlock(512, 512))
        self.up2 = nn.Sequential(UpscaleSubpixel(512, 256, leaky_slope=0.2),
                                 ResidualBlock(256, 256))
        self.up3 = nn.Sequential(UpscaleSubpixel(256, 128, leaky_slope=0.2),
                                 ResidualBlock(128, 128))
        self.conv = nn.Conv2d(128, 3, 5, stride=1, padding=2)

        if learn_mask:
            self.mask_up1 = UpscaleSubpixel(512, 512)
            self.mask_up2 = UpscaleSubpixel(512, 256)
            self.mask_up3 = UpscaleSubpixel(256, 128)
            self.mask_conv = nn.Conv2d(128, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass through the Villain Faceswap decoder

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


class Villain(ModelPlugin):
    """ Villain Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    """
    def __init__(self, num_identities: int = 2) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(num_identities, input_size=128)
        self.encoder = Encoder(cfg.lowmem())
        self.decoders = nn.ModuleList(Decoder(cfg_loss.learn_mask())
                                      for _ in range(num_identities))

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[tuple[torch.Tensor, ...]]:
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
        encoded = [self.encoder(x) for x in inputs]
        decoded = tuple(dec(x) for dec, x in zip(self.decoders, encoded))
        return decoded


__all__ = get_module_objects(__name__)


if __name__ == "__main__":
    p = Villain(2)
    i = [torch.rand((1, 3, 128, 128)), torch.rand((1, 3, 128, 128))]
    out = p(i)
    print([[k.shape for k in j] for j in out])

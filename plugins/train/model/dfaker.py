#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """
from __future__ import annotations

import logging
import sys
from collections import OrderedDict

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers_legacy import ConvBlockLegacy
from lib.model.nn_blocks import UpscaleSubpixel, ResidualBlock
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss

from .base import ModelPlugin
from . import dfaker_defaults as cfg

logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Encoder(nn.Module):
    """ The DFaker Encoder

    Parameters
    ----------
    input_size
        The pixel input size to the model
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self, input_size: int, is_legacy: bool) -> None:
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
        dim = 4 if input_size == 64 else 8
        self.dense1 = nn.Linear(1024 * dim * dim, 1024)
        self.dense2 = nn.Linear(1024, 1024 * 4 * 4)
        self.up = UpscaleSubpixel(1024, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the DFaker encoder

        Parameters
        ----------
        inputs
            The input to the encoder

        Returns
        -------
        The output from the encoder
        """
        x: torch.Tensor = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        return self.up(x)


class Decoder(nn.Module):
    """ The DFaker Decoder Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self, learn_mask: bool, output_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        ins = [512, 1024, 512, 256, 128]
        outs = [1024, 512, 256, 128, 64]
        if output_size == 128:
            ins[1] = 512
            ins = ins[1:]
            outs = outs[1:]

        up = [nn.Sequential(OrderedDict({"up": UpscaleSubpixel(i, o, leaky_slope=-1.),
                                         "act": nn.LeakyReLU(negative_slope=0.2),
                                         "res": ResidualBlock(o)}))
              for i, o in zip(ins[:-1], outs[:-1])]
        self.up = nn.Sequential(*up, UpscaleSubpixel(ins[-1], outs[-1]))
        self.conv = nn.Conv2d(64, 3, 5, stride=1, padding=2)

        self.mask_up = None
        if learn_mask:
            self.mask_up = nn.Sequential(*(UpscaleSubpixel(i, o) for i, o in zip(ins, outs)))
            self.mask_conv = nn.Conv2d(64, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the DFaker decoder

        Parameters
        ----------
        inputs
            The input to the Decoder

        Returns
        -------
        outputs
            The image output and optionally mask from the decoder
        """
        x = self.up(inputs)
        x = torch.sigmoid(self.conv(x))

        if self.mask_up is None:
            return (x, )

        mask = self.mask_up(inputs)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class DFaker(ModelPlugin):
    """ Dfaker Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, num_identities: int = 2, is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        output_size = cfg.output_size()
        if output_size not in (128, 256):
            logger.error("Dfaker output shape should be 128 or 256 px")
            sys.exit(1)
        super().__init__(num_identities, input_size=output_size // 2, is_legacy=is_legacy)
        self.encoder = Encoder(self.input_shape[1], self.is_legacy)
        self.decoders = nn.ModuleList(Decoder(cfg_loss.learn_mask(), output_size)
                                      for _ in range(num_identities))

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...]]:
        """ Forward pass through the DFaker model

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

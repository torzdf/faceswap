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
from lib.model.nn_blocks import UpscaleSubpixel, ResidualBlock
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss
from .base import ModelPlugin

from .original import Encoder
from . import dfaker_defaults as cfg

logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Decoder(nn.Module):
    """The DFaker Decoder Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self, learn_mask: bool, output_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        ins = [512, 512, 512, 256, 128]
        outs = [1024, 512, 256, 128, 64]
        if output_size == 128:
            ins = ins[1:]
            outs = outs[1:]
        self.upscale = nn.Sequential(
            *(nn.Sequential(OrderedDict({"up": UpscaleSubpixel(i, o),
                                         "act": nn.LeakyReLU(negative_slope=0.2),
                                         "res": ResidualBlock(o, o, padding=1)}))
              for i, o in zip(ins, outs))
        )
        self.conv = nn.Conv2d(64, 3, 5, stride=1, padding=2)

        self.upscale_mask = None
        if learn_mask:
            self.upscale_mask = nn.Sequential(*(UpscaleSubpixel(i, o) for i, o in zip(ins, outs)))
            self.conv_mask = nn.Conv2d(64, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass through the DFaker decoder

        Parameters
        ----------
        inputs
            The input to the Decoder

        Returns
        -------
        outputs
            The image output and optionally mask from the decoder
        """
        x = self.upscale(inputs)
        x = torch.sigmoid(self.conv(x))

        if self.upscale_mask is None:
            return (x, )

        mask = self.upscale_mask(inputs)
        mask = torch.sigmoid(self.conv_mask(mask))
        return (x, mask)


class DFaker(ModelPlugin):
    """ Dfaker Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    """
    def __init__(self, num_identities: int = 2) -> None:
        logger.debug(parse_class_init(locals()))

        output_size = cfg.output_size()
        if output_size not in (128, 256):
            logger.error("Dfaker output shape should be 128 or 256 px")
            sys.exit(1)
        super().__init__(num_identities, input_size=output_size // 2)
        self.encoder = Encoder(low_mem=False)
        self.decoders = nn.ModuleList(Decoder(cfg_loss.learn_mask(), output_size)
                                      for _ in range(num_identities))

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...]]:
        """Forward pass through the DFaker model

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
    # TODO validate and remove test code
    p = DFaker(2)
    t = [torch.rand((1, 3, 64, 64)), torch.rand((1, 3, 64, 64))]
    p(t)

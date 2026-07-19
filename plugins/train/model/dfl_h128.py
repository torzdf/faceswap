#!/usr/bin/env python3
""" DeepFaceLab H128 Model
    Based on https://github.com/iperov/DeepFaceLab
"""
import logging

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers import Reshape, UpscaleSubpixel
from lib.model.layers_legacy import Conv2dLegacy
from lib.utils import get_module_objects
from plugins.train.train_config import Loss as cfg_loss

from .base import ModelPlugin
from . import dfl_h128_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Encoder(nn.Sequential):  # pylint:disable=too-many-instance-attributes
    """ The DFL-H128 Encoder

    Parameters
    ----------
    encoder_dim
        The size of the bottleneck and subsequent multiplier
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, encoder_dim: int, is_legacy: bool) -> None:
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
        self.conv4 = conv(512, 1024, 5, stride=2, padding=padding)
        self.act4 = nn.LeakyReLU(0.1, inplace=True)

        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(1024 * 8 * 8, encoder_dim)
        self.dense2 = nn.Linear(encoder_dim, encoder_dim * 8 * 8)
        self.reshape = Reshape((encoder_dim, 8, 8), is_contiguous=True)

        self.up = UpscaleSubpixel(encoder_dim, encoder_dim)


class Decoder(nn.Module):
    """ The DFL-H128 Faceswap Decoder Network.

    Parameters
    ----------
    encoder_dim
        The size of the bottleneck and subsequent multiplier
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self, encoder_dim: int, learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.up1 = UpscaleSubpixel(encoder_dim, encoder_dim)
        self.up2 = UpscaleSubpixel(encoder_dim, encoder_dim // 2)
        self.up3 = UpscaleSubpixel(encoder_dim // 2, encoder_dim // 4)
        self.conv = nn.Conv2d(encoder_dim // 4, 3, 5, stride=1, padding=2)

        self.mask_up1 = None
        if learn_mask:
            self.mask_up1 = UpscaleSubpixel(encoder_dim, encoder_dim)
            self.mask_up2 = UpscaleSubpixel(encoder_dim, encoder_dim // 2)
            self.mask_up3 = UpscaleSubpixel(encoder_dim // 2, encoder_dim // 4)
            self.mask_conv = nn.Conv2d(encoder_dim // 4, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the DFL-H128 decoder

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

        if self.mask_up1 is None:
            return (x, )

        mask = self.mask_up1(inputs)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class DflH128(ModelPlugin):
    """ DFL-H128 Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, num_identities: int = 2, is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(num_identities, input_size=128, is_legacy=is_legacy)
        encoder_dim = 256 if cfg.lowmem() else 512
        self.encoder = Encoder(encoder_dim, self.is_legacy)
        self.decoders = nn.ModuleList(Decoder(encoder_dim, cfg_loss.learn_mask())
                                      for _ in range(num_identities))

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...]]:
        """ Forward pass through the DFL-H128 model

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

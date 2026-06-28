#!/usr/bin/env python3
""" A lightweight variant of DFaker Model
    By AnDenix, 2018-2019
    Based on the dfaker model: https://github.com/dfaker

    Acknowledgments:
    kvrooman for numerous insights and invaluable aid
    DeepHomage for lots of testing
    """
from __future__ import annotations

import logging
import typing as T
from collections import OrderedDict

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers import ResidualBlock, UpscaleSubpixel, Upscale2xBlock
from lib.model.layers_legacy import ConvBlockLegacy, UpSampling2dLegacy
from lib.utils import FaceswapError, get_module_objects
from plugins.train.train_config import Loss as cfg_loss

from .base import ModelPlugin
from . import dlight_defaults as cfg


logger = logging.getLogger(__name__)


class Encoder(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Dlight Encoder

    Parameters
    ----------
    encoder_filters
        The base filters to use for each convolution
    encoder_dim
        The bottleneck size
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self, encoder_filters: int, encoder_dim: int, is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        in_chan = 3
        out_chan = encoder_filters // 2
        if is_legacy:
            self.conv1 = ConvBlockLegacy(in_chan, out_chan, 5, stride=2, padding="same")
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))

        self.pool1 = nn.AvgPool2d((2, 2))
        self.leaky1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        in_chan += out_chan
        out_chan *= 2
        if is_legacy:
            self.conv2 = ConvBlockLegacy(in_chan, out_chan, 5, stride=2, padding="same")
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))

        self.pool2 = nn.AvgPool2d((2, 2))
        self.leaky2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        in_chan += out_chan
        out_chan *= 2
        if is_legacy:
            self.conv3 = ConvBlockLegacy(in_chan, out_chan, 5, stride=2, padding="same")
        else:
            self.conv3 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))

        self.pool3 = nn.AvgPool2d((2, 2))
        self.leaky3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        in_chan += out_chan
        out_chan *= 2
        if is_legacy:
            self.conv4 = ConvBlockLegacy(in_chan, out_chan, 5, stride=2, padding="same")
        else:
            self.conv4 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))

        self.pool4 = nn.AvgPool2d((2, 2))
        self.leaky4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        in_chan += out_chan
        out_chan *= 2
        if is_legacy:
            self.conv5 = ConvBlockLegacy(in_chan, out_chan, 5, stride=2, padding="same")
        else:
            self.conv5 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 5, stride=2, padding=2),
                                       nn.LeakyReLU(0.1, inplace=True))

        self.pool5 = nn.AvgPool2d((2, 2))
        self.leaky5 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        in_chan += out_chan
        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(in_chan * 4 * 4, encoder_dim)
        self.drop1 = nn.Dropout(p=0.05, inplace=True)
        self.dense2 = nn.Linear(encoder_dim, 1024 * 4 * 4)
        self.drop2 = nn.Dropout(p=0.05, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Dlight encoder

        Parameters
        ----------
        inputs
            The input to the encoder

        Returns
        -------
        The output from the encoder
        """
        x1 = self.conv1(inputs)
        x2 = self.pool1(inputs)
        x2 = self.leaky1(x2)
        x = torch.concat((x1, x2), dim=1)

        x1 = self.conv2(x)
        x2 = self.pool2(x)
        x2 = self.leaky2(x2)
        x = torch.concat((x1, x2), dim=1)

        x1 = self.conv3(x)
        x2 = self.pool3(x)
        x2 = self.leaky3(x2)
        x = torch.concat((x1, x2), dim=1)

        x1 = self.conv4(x)
        x2 = self.pool4(x)
        x2 = self.leaky4(x2)
        x = torch.concat((x1, x2), dim=1)

        x1 = self.conv5(x)
        x2 = self.pool5(x)
        x2 = self.leaky5(x2)
        x = torch.concat((x1, x2), dim=1)

        x = self.drop1(self.dense1(self.flatten(x)))
        x = T.cast(torch.Tensor, self.drop2(self.dense2(x)))
        return x.view(x.shape[0], 1024, 4, 4)


class DecoderA(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Dlight Faceswap Decoder A Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    upscale_ratio
        The amount to upscale the input to the layer
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self, learn_mask: bool, upscale_ratio: int, is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask

        dec_a_complexity = 256
        mask_complexity = 128

        if is_legacy:
            self.up1 = UpSampling2dLegacy(size=upscale_ratio, interpolation="bilinear")
        else:
            self.up1 = nn.UpsamplingBilinear2d(scale_factor=upscale_ratio)

        self.up2 = Upscale2xBlock(1024, dec_a_complexity, fast=False, is_legacy=is_legacy)
        self.up3 = Upscale2xBlock(dec_a_complexity,
                                  dec_a_complexity // 2,
                                  fast=False,
                                  is_legacy=is_legacy)
        self.up4 = Upscale2xBlock(dec_a_complexity // 2,
                                  dec_a_complexity // 4,
                                  fast=False,
                                  is_legacy=is_legacy)
        self.up5 = Upscale2xBlock(dec_a_complexity // 4,
                                  dec_a_complexity // 8,
                                  fast=False,
                                  is_legacy=is_legacy)
        self.conv = nn.Conv2d(dec_a_complexity // 8, 3, 5, stride=1, padding=2)

        if self.learn_mask:
            self.mask_up1 = Upscale2xBlock(1024, mask_complexity, fast=False, is_legacy=is_legacy)
            self.mask_up2 = Upscale2xBlock(mask_complexity,
                                           mask_complexity // 2,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_up3 = Upscale2xBlock(mask_complexity // 2,
                                           mask_complexity // 4,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_up4 = Upscale2xBlock(mask_complexity // 4,
                                           mask_complexity // 8,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_conv = nn.Conv2d(mask_complexity // 8, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the Dlight decoder A

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
        xy = x
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = torch.sigmoid(self.conv(x))

        if not self.learn_mask:
            return (x, )

        mask = self.mask_up1(xy)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = self.mask_up4(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class DecoderB(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Dlight Faceswap Decoder B Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    upscale_ratio
        The amount to upscale the input to the layer
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self, learn_mask: bool, upscale_ratio: int, is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask

        dec_b_complexity = 512
        mask_complexity = 128

        self.up1 = Upscale2xBlock(1024,
                                  dec_b_complexity,
                                  scale_factor=upscale_ratio,
                                  fast=False,
                                  activation=False,
                                  is_legacy=is_legacy)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.res1 = nn.Sequential(OrderedDict({
            "res1": ResidualBlock(dec_b_complexity, bias=True),
            "res2": ResidualBlock(dec_b_complexity, bias=False),
            "res3": ResidualBlock(dec_b_complexity, bias=False)
        }))

        self.up2 = nn.Sequential(OrderedDict({
            "up": Upscale2xBlock(dec_b_complexity,
                                 dec_b_complexity,
                                 fast=False,
                                 activation=False,
                                 is_legacy=is_legacy),
            "act": nn.LeakyReLU(negative_slope=0.2, inplace=True),
            "res1": ResidualBlock(dec_b_complexity, bias=True),
            "res2": ResidualBlock(dec_b_complexity, bias=False),
            "bn": nn.BatchNorm2d(dec_b_complexity, eps=0.001, momentum=0.01)
        }))

        self.up3 = nn.Sequential(OrderedDict({
            "up": Upscale2xBlock(dec_b_complexity,
                                 dec_b_complexity // 2,
                                 fast=False,
                                 activation=False,
                                 is_legacy=is_legacy),
            "act": nn.LeakyReLU(negative_slope=0.2, inplace=True),
            "res": ResidualBlock(dec_b_complexity // 2, padding=1, bias=True)
        }))

        self.up4 = nn.Sequential(OrderedDict({
            "up": Upscale2xBlock(dec_b_complexity // 2,
                                 dec_b_complexity // 4,
                                 fast=False,
                                 activation=False,
                                 is_legacy=is_legacy),
            "act": nn.LeakyReLU(negative_slope=0.2, inplace=True),
            "res": ResidualBlock(dec_b_complexity // 4, padding=1, bias=False),
            "bn": nn.BatchNorm2d(dec_b_complexity // 4, eps=0.001, momentum=0.01)
        }))

        self.up5 = Upscale2xBlock(dec_b_complexity // 4,
                                  dec_b_complexity // 8,
                                  fast=False,
                                  activation=True,
                                  is_legacy=is_legacy)
        self.conv = nn.Conv2d(dec_b_complexity // 8, 3, 5, stride=1, padding=2)

        if self.learn_mask:
            self.mask_up1 = Upscale2xBlock(512, mask_complexity, fast=False, is_legacy=is_legacy)
            self.mask_up2 = Upscale2xBlock(mask_complexity,
                                           mask_complexity // 2,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_up3 = Upscale2xBlock(mask_complexity // 2,
                                           mask_complexity // 4,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_up4 = Upscale2xBlock(mask_complexity // 4,
                                           mask_complexity // 8,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_conv = nn.Conv2d(mask_complexity // 8, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the Dlight decoder B

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
        xy = x
        x = self.leaky1(x)
        x = self.res1(x)

        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = torch.sigmoid(self.conv(x))

        if not self.learn_mask:
            return (x, )

        mask = self.mask_up1(xy)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = self.mask_up4(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class DecoderBFast(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Dlight Faceswap Decoder B Fast Network.

    Parameters
    ----------
    learn_mask
        ``True`` to set a secondary task to learn a mask
    upscale_ratio
        The amount to upscale the input to the layer
    is_legacy
        ``True`` if the model was originally created in Keras
    """
    def __init__(self, learn_mask: bool, upscale_ratio: int, is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask

        dec_b_complexity = 512
        mask_complexity = 128

        self.up1 = UpscaleSubpixel(1024, dec_b_complexity, scale_factor=upscale_ratio)
        self.up2 = Upscale2xBlock(dec_b_complexity,
                                  dec_b_complexity,
                                  fast=True,
                                  is_legacy=is_legacy)
        self.up3 = Upscale2xBlock(dec_b_complexity,
                                  dec_b_complexity // 2,
                                  fast=True,
                                  is_legacy=is_legacy)
        self.up4 = Upscale2xBlock(dec_b_complexity // 2,
                                  dec_b_complexity // 4,
                                  fast=True,
                                  is_legacy=is_legacy)
        self.up5 = Upscale2xBlock(dec_b_complexity // 4,
                                  dec_b_complexity // 8,
                                  fast=True,
                                  is_legacy=is_legacy)

        self.conv = nn.Conv2d(dec_b_complexity // 8, 3, 5, stride=1, padding=2)

        if self.learn_mask:
            self.mask_up1 = Upscale2xBlock(512, mask_complexity, fast=False, is_legacy=is_legacy)
            self.mask_up2 = Upscale2xBlock(mask_complexity,
                                           mask_complexity // 2,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_up3 = Upscale2xBlock(mask_complexity // 2,
                                           mask_complexity // 4,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_up4 = Upscale2xBlock(mask_complexity // 4,
                                           mask_complexity // 8,
                                           fast=False,
                                           is_legacy=is_legacy)
            self.mask_conv = nn.Conv2d(mask_complexity // 8, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the Dlight B Fast decoder

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
        xy = x
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = torch.sigmoid(self.conv(x))

        if not self.learn_mask:
            return (x, )

        mask = self.mask_up1(xy)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = self.mask_up4(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class Dlight(ModelPlugin):
    """ Dlight Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, num_identities: int = 2, is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        if num_identities != 2:
            raise FaceswapError(f"{self.__class__.__name__} only supports 2 identities. Reduce "
                                "the number of identities or choose a different model")
        super().__init__(num_identities, input_size=128, is_legacy=is_legacy)

        learn_mask = cfg_loss.learn_mask()
        features = {"lowmem": 0, "fair": 1, "best": 2}[cfg.features()]
        details = {"fast": 0, "good": 1}[cfg.details()]

        up_ratios = {128: 2, 256: 4, 384: 6}
        out_size = cfg.output_size()
        if out_size not in up_ratios:
            raise FaceswapError("Config error: output_size must be one of: 128, 256, or 384.")
        upscale_ratio = up_ratios[out_size]

        encoder_filters = 64 if features > 0 else 48
        bonum_fortunam = 128
        encoder_dim = {0: 512 + bonum_fortunam,
                       1: 1024 + bonum_fortunam,
                       2: 1536 + bonum_fortunam}[features]

        dec_b = DecoderB if details > 0 else DecoderBFast
        self.encoder = Encoder(encoder_filters, encoder_dim, self.is_legacy)
        self.decoders = nn.ModuleList((DecoderA(learn_mask, upscale_ratio, self.is_legacy),
                                       dec_b(learn_mask, upscale_ratio, self.is_legacy)))

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[tuple[torch.Tensor, ...]]:
        """ Forward pass through the Dlight model

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

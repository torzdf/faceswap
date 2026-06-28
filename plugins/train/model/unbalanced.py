#!/usr/bin/env python3
""" Unbalanced Model
    Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contributions """
from __future__ import annotations

import logging

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers import ResidualBlock, UpscaleSubpixel
from lib.model.layers_legacy import ConvBlockLegacy, InstanceNormLegacy
from lib.utils import FaceswapError, get_module_objects
from plugins.train.train_config import Loss as cfg_loss
from .base import ModelPlugin

from . import unbalanced_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Encoder(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Unbalanced Encoder

    Parameters
    ----------
    complexity
        Encoder Convolution Layer Complexity
    bottleneck
        The number of nodes in the bottleneck
    dense_dim
        The dimensions to reshape the bottleneck
    input_size
        The pixel input dimension to the encoder
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self,
                 complexity: int,
                 bottleneck: int,
                 dense_dim: int,
                 input_size: int,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.dense_dim = dense_dim
        self.dense_width = input_size // 16
        half_width = self.dense_width // 2

        if is_legacy:
            self.down1 = ConvBlockLegacy(3, complexity, 5, stride=2, leaky_slope=-1.)
            self.norm1 = InstanceNormLegacy()
        else:
            self.down1 = nn.Conv2d(3, complexity, 5, stride=2, padding=2)
            self.norm1 = nn.InstanceNorm2d(complexity, affine=True)
        self.leaky1 = nn.LeakyReLU(0.1, inplace=True)

        if is_legacy:
            self.down2 = ConvBlockLegacy(complexity, complexity * 2, 5, stride=2, leaky_slope=-1.)
            self.norm2 = InstanceNormLegacy()
        else:
            self.down2 = nn.Conv2d(complexity, complexity * 2, 5, stride=2, padding=2)
            self.norm2 = nn.InstanceNorm2d(complexity * 2, affine=True)
        self.leaky2 = nn.LeakyReLU(0.1, inplace=True)

        if is_legacy:
            self.down3 = ConvBlockLegacy(complexity * 2, complexity * 4, 5, stride=2)
            self.down4 = ConvBlockLegacy(complexity * 4, complexity * 6, 5, stride=2)
            self.down5 = ConvBlockLegacy(complexity * 6, complexity * 8, 5, stride=2)
        else:
            self.down3 = nn.Sequential(
                nn.Conv2d(complexity * 2, complexity * 4, 5, stride=2, padding=2),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.down4 = nn.Sequential(
                nn.Conv2d(complexity * 4, complexity * 6, 5, stride=2, padding=2),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.down5 = nn.Sequential(
                nn.Conv2d(complexity * 6, complexity * 8, 5, stride=2, padding=2),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(complexity * 8 * half_width * half_width, bottleneck)
        self.dense2 = nn.Linear(bottleneck, self.dense_dim * self.dense_width * self.dense_width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Original encoder

        Parameters
        ----------
        inputs
            The input to the encoder

        Returns
        -------
        The output from the encoder
        """
        x = self.down1(inputs)
        x = self.norm1(x)
        x = self.leaky1(x)
        x = self.down2(x)
        x = self.norm2(x)
        x = self.leaky2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x: torch.Tensor = self.dense2(x)
        return x.view(x.shape[0], self.dense_dim, self.dense_width, self.dense_width)


class DecoderA(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Faceswap Unbalanced Decoder A Network.

    Parameters
    ----------
    in_channels
        The number of input channels to the first upscale
    complexity
        Decoder A Convolution Layer Complexity
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self,
                 in_channels: int,
                 complexity: int,
                 learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask

        self.up1 = nn.Sequential(UpscaleSubpixel(in_channels, complexity, kernel_size=5),
                                 nn.Dropout(0.25, inplace=True))
        self.up2 = nn.Sequential(UpscaleSubpixel(complexity, complexity, kernel_size=5),
                                 nn.Dropout(0.15 if in_channels < 512 else 0.25, inplace=True))
        self.up3 = UpscaleSubpixel(complexity, complexity // 2, kernel_size=5)
        self.up4 = UpscaleSubpixel(complexity // 2, complexity // 4, kernel_size=5)
        self.conv = nn.Conv2d(complexity // 4, 3, 5, stride=1, padding=2)
        self.act = nn.Sigmoid()

        if self.learn_mask:
            self.mask_up1 = UpscaleSubpixel(in_channels, complexity)
            self.mask_up2 = UpscaleSubpixel(complexity, complexity)
            self.mask_up3 = UpscaleSubpixel(complexity, complexity // 2)
            self.mask_up4 = UpscaleSubpixel(complexity // 2, complexity // 4)
            self.mask_conv = nn.Conv2d(complexity // 4, 1, 5, stride=1, padding=2)
            self.mask_act = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the RealFace A decoder

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
        x = self.conv(x)
        x = self.act(x)
        if not self.learn_mask:
            return (x, )

        mask = self.mask_up1(inputs)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = self.mask_up4(mask)
        mask = self.mask_conv(mask)
        mask = self.mask_act(mask)
        return (x, mask)


class DecoderB(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Faceswap Unbalanced Decoder B Network.

    Parameters
    ----------
    in_channels
        The number of input channels to the first upscale
    complexity
        Encoder Convolution Layer Complexity
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self,
                 in_channels: int,
                 complexity: int,
                 learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask
        low_mem = in_channels < 512

        if low_mem:
            channels = [complexity, complexity // 2, complexity // 4, complexity // 8]
            slope = 0.1
        else:
            channels = [complexity, complexity, complexity // 2, complexity // 4]
            slope = 0.2

        self.up1 = UpscaleSubpixel(in_channels, channels[0], kernel_size=5, leaky_slope=slope)
        self.up2 = UpscaleSubpixel(channels[0], channels[1], kernel_size=5, leaky_slope=slope)
        self.up3 = UpscaleSubpixel(channels[1], channels[2], kernel_size=5, leaky_slope=slope)
        self.up4 = UpscaleSubpixel(channels[2], channels[3], kernel_size=5, leaky_slope=0.1)

        if not low_mem:
            self.up1 = nn.Sequential(self.up1, ResidualBlock(channels[0]))
            self.up2 = nn.Sequential(self.up2, ResidualBlock(channels[1]))
            self.up3 = nn.Sequential(self.up3, ResidualBlock(channels[2]))
        self.conv = nn.Conv2d(channels[3], 3, 5, stride=1, padding=2)
        self.act = nn.Sigmoid()

        if self.learn_mask:
            self.mask_up1 = UpscaleSubpixel(in_channels, channels[0])
            self.mask_up2 = UpscaleSubpixel(channels[0], channels[1])
            self.mask_up3 = UpscaleSubpixel(channels[1], channels[2])
            self.mask_up4 = UpscaleSubpixel(channels[2], channels[3])
            self.mask_conv = nn.Conv2d(channels[3], 1, 5, stride=1, padding=2)
            self.mask_act = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the RealFace A decoder

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
        x = self.conv(x)
        x = self.act(x)
        if not self.learn_mask:
            return (x, )

        mask = self.mask_up1(inputs)
        mask = self.mask_up2(mask)
        mask = self.mask_up3(mask)
        mask = self.mask_up4(mask)
        mask = self.mask_conv(mask)
        mask = self.mask_act(mask)
        return (x, mask)


class Unbalanced(ModelPlugin):
    """ Unbalanced Faceswap Model.

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
        super().__init__(num_identities, input_size=cfg.input_size(), is_legacy=is_legacy)

        dense_dim = 384 if cfg.lowmem() else 512
        self.encoder = Encoder(128 if cfg.lowmem() else cfg.complexity_encoder(),
                               512 if cfg.lowmem() else cfg.nodes(),
                               dense_dim,
                               self.input_shape[-1],
                               self.is_legacy)
        self.decoder_a = DecoderA(dense_dim,
                                  320 if cfg.lowmem() else cfg.complexity_decoder_a(),
                                  cfg_loss.learn_mask())
        self.decoder_b = DecoderB(dense_dim,
                                  384 if cfg.lowmem() else cfg.complexity_decoder_b(),
                                  cfg_loss.learn_mask())

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
        decoded = tuple(dec(x) for dec, x in zip((self.decoder_a, self.decoder_b), encoded))
        return decoded


__all__ = get_module_objects(__name__)

#!/usr/bin/env python3
""" RealFaceRC1, codenamed 'Pegasus'
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contributions
    Major thanks goes to BryanLyon as it vastly powered by his ideas and insights.
    Without him it would not be possible to come up with the model.
    Additional thanks: Birb - source of inspiration, great Encoder ideas
                       Kvrooman - additional counseling on auto-encoders and practical advice
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers_legacy import ConvBlockLegacy
from lib.model.nn_blocks import ResidualBlock, UpscaleSubpixel
from lib.utils import FaceswapError, get_module_objects
from plugins.train.train_config import Loss as cfg_loss
from .base import ModelPlugin
from . import realface_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class Encoder(nn.Module):
    """ The RealFace Encoder

    Parameters
    ----------
    complexity
        Encoder Convolution Layer Complexity
    num_downscale
        The number of downscale blocks
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, complexity: int, num_downscale: int = 4, is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        channels = [3] + [complexity * 2 ** i for i in range(num_downscale)]
        if is_legacy:
            blocks: list[nn.Module] = [
                nn.Sequential(ConvBlockLegacy(channels[i], channels[i + 1], 5,
                                              stride=2,
                                              padding="same",
                                              leaky_slope=0.2),
                              ResidualBlock(channels[i + 1], bias=True),
                              ResidualBlock(channels[i + 1], bias=True))
                for i in range(num_downscale - 1)
                ]
            blocks.append(ConvBlockLegacy(channels[-2], channels[-1], 5,
                                          stride=2,
                                          padding="same",
                                          leaky_slope=0.1))
        else:
            blocks = [nn.Sequential(nn.Conv2d(channels[i], channels[i + 1], 5,
                                              stride=2,
                                              padding=2),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    ResidualBlock(channels[i + 1], bias=True),
                                    ResidualBlock(channels[i + 1], bias=True))
                      for i in range(num_downscale - 1)]
            blocks.extend([nn.Conv2d(channels[-2], channels[-1], 5,
                                     stride=2,
                                     padding=2),
                           nn.LeakyReLU(0.1, inplace=True)])
        self.down = nn.Sequential(*blocks)

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
        return self.down(inputs)


class DecoderA(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Faceswap RealFace Decoder A Network.

    Parameters
    ----------
    in_channels
        The number of input channels to the first dense layer
    out_channels
        The number of output channels from the 2nd dense layer (1st pseudo upscale)
    bottleneck_size
        The number of nodes in the bottleneck
    upscale_width
        The pixel dimension of the first upscale
    complexity
        The convolution complexity for decoder A
    num_upscale
        The number of upscale blocks
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bottleneck_size: int,
                 upscale_width: int,
                 complexity: int,
                 num_upscale: int,
                 learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask
        self.out_channels = out_channels
        self.upscale_width = upscale_width

        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(in_channels * upscale_width * upscale_width, bottleneck_size)
        self.dense2 = nn.Linear(bottleneck_size, out_channels * upscale_width * upscale_width)
        self.up1 = UpscaleSubpixel(out_channels, out_channels, leaky_slope=-1.)
        self.leaky = nn.LeakyReLU(0.2)
        self.res = ResidualBlock(out_channels, bias=False)

        channels = [out_channels] + [complexity // 2 ** i for i in range(num_upscale - 1)]
        blocks: list[nn.Module] = [UpscaleSubpixel(channels[i], channels[i + 1])
                                   for i in range(num_upscale - 2)]
        blocks.append(UpscaleSubpixel(channels[-2], channels[-1]))
        self.up2 = nn.Sequential(*blocks)
        self.conv = nn.Conv2d(channels[-1], 3, 5, stride=1, padding=2)

        if self.learn_mask:
            self.leaky_mask = nn.LeakyReLU(0.1)

            m_complexity = 384
            m_channels = [out_channels] + [m_complexity // 2 ** i for i in range(num_upscale - 1)]
            m_blocks: list[nn.Module] = [UpscaleSubpixel(m_channels[i], m_channels[i + 1])
                                         for i in range(num_upscale - 2)]
            m_blocks.append(UpscaleSubpixel(m_channels[-2], m_channels[-1]))
            self.mask_up = nn.Sequential(*m_blocks)
            self.mask_conv = nn.Conv2d(m_channels[-1], 1, 5, stride=1, padding=2)

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
        x = self.flatten(inputs)
        x = self.dense1(x)
        x: torch.Tensor = self.dense2(x)
        x = x.view(x.shape[0], self.out_channels, self.upscale_width, self.upscale_width)
        x = self.up1(x)

        mask = x

        x = self.leaky(x)
        x = self.res(x)
        x = self.up2(x)
        x = torch.sigmoid(self.conv(x))

        if not self.learn_mask:
            return (x, )

        mask = self.leaky_mask(mask)
        mask = self.mask_up(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class DecoderB(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The Faceswap RealFace Decoder B Network.

    Parameters
    ----------
    in_channels
        The number of input channels to the first dense layer
    out_channels
        The number of output channels from the 2nd dense layer (1st pseudo upscale)
    bottleneck_size
        The number of nodes in the bottleneck
    upscale_width
        The pixel dimension of the first upscale
    complexity
        The convolution complexity for decoder A
    num_upscale
        The number of upscale blocks
    learn_mask
        ``True`` to set a secondary task to learn a mask
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bottleneck_size: int,
                 upscale_width: int,
                 complexity: int,
                 num_upscale: int,
                 learn_mask: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.learn_mask = learn_mask
        self.out_channels = out_channels
        self.upscale_width = upscale_width

        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(in_channels * upscale_width * upscale_width, bottleneck_size)
        self.dense2 = nn.Linear(bottleneck_size, out_channels * upscale_width * upscale_width)
        self.up1 = UpscaleSubpixel(out_channels, out_channels, leaky_slope=-1.)
        self.leaky = nn.LeakyReLU(0.2)
        self.res = ResidualBlock(out_channels, bias=False)

        channels = [out_channels] + [complexity // 2 ** i for i in range(num_upscale - 1)]
        blocks: list[nn.Module] = [
            nn.Sequential(UpscaleSubpixel(channels[i], channels[i + 1], leaky_slope=0.2),
                          ResidualBlock(channels[i + 1], bias=False),
                          ResidualBlock(channels[i + 1], bias=True))
            for i in range(num_upscale - 2)
            ]
        blocks.append(UpscaleSubpixel(channels[-2], channels[-1]))
        self.up2 = nn.Sequential(*blocks)
        self.conv = nn.Conv2d(channels[-1], 3, 5, stride=1, padding=2)

        if self.learn_mask:
            self.leaky_mask = nn.LeakyReLU(0.1)

            m_complexity = 384
            m_channels = [out_channels] + [m_complexity // 2 ** i for i in range(num_upscale - 1)]
            m_blocks: list[nn.Module] = [UpscaleSubpixel(m_channels[i], m_channels[i + 1])
                                         for i in range(num_upscale - 2)]
            m_blocks.append(UpscaleSubpixel(m_channels[-2], m_channels[-1]))
            self.mask_up = nn.Sequential(*m_blocks)
            self.mask_conv = nn.Conv2d(m_channels[-1], 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the RealFace B decoder

        Parameters
        ----------
        inputs
            The input to the Decoder

        Returns
        -------
        outputs
            The image output and optionally mask from the decoder
        """
        x = self.flatten(inputs)
        x = self.dense1(x)
        x: torch.Tensor = self.dense2(x)
        x = x.view(x.shape[0], self.out_channels, self.upscale_width, self.upscale_width)
        x = self.up1(x)

        mask = x

        x = self.leaky(x)
        x = self.res(x)
        x = self.up2(x)
        x = torch.sigmoid(self.conv(x))

        if not self.learn_mask:
            return (x, )

        mask = self.leaky_mask(mask)
        mask = self.mask_up(mask)
        mask = torch.sigmoid(self.mask_conv(mask))
        return (x, mask)


class RealFace(ModelPlugin):
    """ RealFace Faceswap Model.

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
        input_size = cfg.input_size()
        if not 64 <= input_size <= 128 or input_size % 16 != 0:
            raise FaceswapError(f"Config error: {self.__class__.__name__} input_size must be "
                                "between 64 and 128 and be divisible by 16.")
        num_downscale = 4
        downscale_ratio = 2 ** num_downscale
        dense_width, num_upscale = self._get_dense_width_upscaler_numbers(input_size,
                                                                          downscale_ratio)
        dense_filters = (int(1024 - (dense_width - 4) * 64) // 16) * 16
        super().__init__(num_identities, input_size=input_size, is_legacy=is_legacy)

        self.encoder = Encoder(cfg.complexity_encoder(),
                               num_downscale=num_downscale,
                               is_legacy=self.is_legacy)

        dec_input_filters = cfg.complexity_encoder() * 2 ** (num_downscale - 1)
        dec_upscale_width = cfg.input_size() // downscale_ratio
        self.decoder_a = DecoderA(dec_input_filters,
                                  int(dense_filters / 1.5),
                                  int(cfg.dense_nodes() / 1.5),
                                  dec_upscale_width,
                                  int(cfg.complexity_decoder() / 1.5),
                                  num_upscale,
                                  cfg_loss.learn_mask())
        self.decoder_b = DecoderB(dec_input_filters,
                                  dense_filters,
                                  cfg.dense_nodes(),
                                  dec_upscale_width,
                                  cfg.complexity_decoder(),
                                  num_upscale,
                                  cfg_loss.learn_mask())

    def _get_dense_width_upscaler_numbers(self, input_size: int, downscale_ratio: int
                                          ) -> tuple[int, int]:
        """ Return the dense width and number of upscale blocks

        Parameters
        ----------
        input_size
            The pixel dimensions of the model input
        downscale_ratio
            The ratio for downscale

        Returns
        -------
        dense_width
            The pixel dimension of the first upscale in the decoder
        num_upscale
            The upscale count of the decoder
        """
        output_size = cfg.output_size()
        if not 64 <= output_size <= 256 or output_size % 32 != 0:
            raise FaceswapError(f"Config error: {self.__class__.__name__} input_size must be "
                                "between 64 and 256 and be divisible by 32.")
        sides = [(output_size // 2 ** n, n) for n in [4, 5] if (output_size // 2 ** n) < 10]
        closest = min((x * downscale_ratio for x, _ in sides),
                      key=lambda x: abs(x - input_size))
        dense_width, num_upscale = [(s, n) for s, n in sides
                                    if s * downscale_ratio == closest][0]
        logger.debug("dense_width: %s, num_upscale: %s", dense_width, num_upscale)
        return dense_width, num_upscale

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

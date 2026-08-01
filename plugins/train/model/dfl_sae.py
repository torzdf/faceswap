#!/usr/bin/env python3
""" DeepFaceLab SAE Model Based on https://github.com/iperov/DeepFaceLab"""
from __future__ import annotations

import logging

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.layers import Reshape, ResidualBlock, UpscaleSubpixel
from lib.model.layers_legacy import Conv2dLegacy
from lib.utils import FaceswapError, get_module_objects
from plugins.train.train_config import Loss as cfg_loss

from .base import ModelPlugin
from . import dfl_sae_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class EncoderDF(nn.Sequential):  # pylint:disable=too-many-instance-attributes
    """ The DeepFaceLab SAE-DF Encoder

    Parameters
    ----------
    input_shape
        The (C, H, W) input shape to the model
    encoder_dim
        The number of dimensions per encoder channel
    ae_dims
        The number of dimensions for the latent space
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 encoder_dim: int,
                 ae_dims: int,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        channels, res = input_shape[:2]
        dims = channels * encoder_dim
        lowest_res = res // 16

        conv = Conv2dLegacy if is_legacy else nn.Conv2d
        padding = "same" if is_legacy else 2

        self.conv1 = conv(channels, dims, 5, stride=2, padding=padding)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv(dims, dims * 2, 5, stride=2, padding=padding)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = conv(dims * 2, dims * 4, 5, stride=2, padding=padding)
        self.act3 = nn.LeakyReLU(0.1, inplace=True)
        self.conv4 = conv(dims * 4, dims * 8, 5, stride=2, padding=padding)
        self.act4 = nn.LeakyReLU(0.1, inplace=True)

        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(dims * 8 * lowest_res * lowest_res, ae_dims)
        self.dense2 = nn.Linear(ae_dims, ae_dims * lowest_res * lowest_res)
        self.reshape = Reshape((ae_dims, lowest_res, lowest_res), is_contiguous=True)
        self.up = UpscaleSubpixel(ae_dims, ae_dims)


class EncoderLIAE(nn.Sequential):
    """ The DeepFaceLab SAE-LIAE Encoder

    Parameters
    ----------
    input_shape
        The (C, H, W) input shape to the model
    encoder_dim
        The number of dimensions per encoder channel
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 encoder_dim: int,
                 is_legacy: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        channels = input_shape[0]
        dims = channels * encoder_dim

        conv = Conv2dLegacy if is_legacy else nn.Conv2d
        padding = "same" if is_legacy else 2

        self.conv1 = conv(channels, dims, 5, stride=2, padding=padding)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv(dims, dims * 2, 5, stride=2, padding=padding)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = conv(dims * 2, dims * 4, 5, stride=2, padding=padding)
        self.act3 = nn.LeakyReLU(0.1, inplace=True)
        self.conv4 = conv(dims * 4, dims * 8, 5, stride=2, padding=padding)
        self.act4 = nn.LeakyReLU(0.1, inplace=True)
        self.flatten = nn.Flatten(start_dim=1)


class InterLIAE(nn.Sequential):
    """ The DeepFaceLab SAE-LIAE Intermediate layer

    Parameters
    ----------
    input_size
        The (C, H, W) shape of the unflattened input tensor
    ae_dims
        The number of dimensions for the latent space
    """
    def __init__(self,
                 input_size: tuple[int, int, int],
                 ae_dims: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        self.dense1 = nn.Linear(input_size[0] * input_size[1] * input_size[1], ae_dims)
        self.dense2 = nn.Linear(ae_dims, ae_dims * 2 * input_size[1] * input_size[1])
        self.reshape = Reshape((ae_dims * 2, input_size[1], input_size[1]), is_contiguous=True)
        self.up = UpscaleSubpixel(ae_dims * 2, ae_dims * 2)


class Decoder(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ The original DeepFaceLab SAE Decoder Network.

    Parameters
    ----------
    in_channels
        The number of input channels to the first convolution
    learn_mask
        ``True`` to set a secondary task to learn a mask
    decoder_dim
        The single channel decoder dimension
    multiscale_count
        The number of multi-scale outputs from the decoder
    """
    def __init__(self,
                 in_channels: int,
                 learn_mask: bool,
                 decoder_dim: int,
                 multiscale_count: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        dims = decoder_dim * 3
        self._multiscale_count = multiscale_count
        self._learn_mask = learn_mask

        self.up1 = UpscaleSubpixel(in_channels, dims * 8, leaky_slope=-1.)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2)
        self.res1_1 = ResidualBlock(dims * 8, padding=1)
        self.res1_2 = ResidualBlock(dims * 8, padding=1)
        if multiscale_count >= 3:
            self.conv_out1 = nn.Conv2d(dims * 8, 3, 5, stride=1, padding=2)

        self.up2 = UpscaleSubpixel(dims * 8, dims * 4, leaky_slope=-1.)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.2)
        self.res2_1 = ResidualBlock(dims * 4, padding=1)
        self.res2_2 = ResidualBlock(dims * 4, padding=1)
        if multiscale_count >= 3:
            self.conv_out2 = nn.Conv2d(dims * 4, 3, 5, stride=1, padding=2)

        self.up3 = UpscaleSubpixel(dims * 4, dims * 2, leaky_slope=-1.)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.2)
        self.res3_1 = ResidualBlock(dims * 2, padding=1)
        self.res3_2 = ResidualBlock(dims * 2, padding=1)

        self.conv_out = nn.Conv2d(dims * 2, 3, 5, stride=1, padding=2)

        if learn_mask:
            self.mask_up1 = UpscaleSubpixel(in_channels, decoder_dim * 8)
            self.mask_up2 = UpscaleSubpixel(decoder_dim * 8, decoder_dim * 4)
            self.mask_up3 = UpscaleSubpixel(decoder_dim * 4, decoder_dim * 2)
            self.mask_conv = nn.Conv2d(decoder_dim * 2, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """ Forward pass through the DeepFaceLab SAE decoder

        Parameters
        ----------
        inputs
            The input to the Decoder

        Returns
        -------
        outputs
            The image output,  at multiple resolutions if multi-scale is enabled, and optionally
            mask from the decoder
        """
        out = []
        x = self.res1_2(self.res1_1(self.leaky1(self.up1(inputs))))
        if self._multiscale_count >= 3:
            out.append(torch.sigmoid(self.conv_out1(x)))

        x = self.res2_2(self.res2_1(self.leaky2(self.up2(x))))
        if self._multiscale_count >= 2:
            out.append(torch.sigmoid(self.conv_out2(x)))

        x = self.res3_2(self.res3_1(self.leaky3(self.up3(x))))
        out.append(torch.sigmoid(self.conv_out(x)))
        if self._learn_mask:
            mask = self.mask_up1(inputs)
            mask = self.mask_up2(mask)
            mask = self.mask_up3(mask)
            out.append(torch.sigmoid(self.mask_conv(mask)))

        return tuple(out)


class DFLSae(ModelPlugin):
    """ DeepFaceLab SAE Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self, num_identities: int = 2, is_legacy: bool = False) -> None:
        if num_identities != 2:
            raise FaceswapError(f"{self.__class__.__name__} only supports 2 identities. Reduce "
                                "the number of identities or choose a different model")
        super().__init__(num_identities, input_size=cfg.input_size(), is_legacy=is_legacy)
        self.architecture = cfg.architecture().lower()

        enc_dim = cfg.encoder_dims()
        ae_dims = cfg.autoencoder_dims()
        if ae_dims == 0:
            ae_dims = 512 if self.architecture == "df" else 256
        dec_dim = cfg.decoder_dims()
        ms_count = 3 if cfg.multiscale_decoder() else 1
        dec_in = ae_dims if self.architecture == "df" else ae_dims * 4

        if self.architecture == "df":
            self.encoder = EncoderDF(self.input_shape, enc_dim, ae_dims, self.is_legacy)
            self.decoders = nn.ModuleList(Decoder(dec_in, cfg_loss.learn_mask(), dec_dim, ms_count)
                                          for _ in range(self.num_identities))
        else:
            int_shape = (enc_dim * 3 * 8, self.input_shape[1] // 16, self.input_shape[1] // 16)
            self.encoder = EncoderLIAE(self.input_shape, enc_dim, self.is_legacy)
            self.inter_both = InterLIAE(int_shape, ae_dims)
            self.inter_side = InterLIAE(int_shape, ae_dims)
            self.decoder = Decoder(dec_in, cfg_loss.learn_mask(), dec_dim, ms_count)

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...]]:
        """ Forward pass through the DeepFaceLab SAE model

        Parameters
        ----------
        inputs: list
            A list of input tensors for the model. This will be of length num_identities with each
            tensor of shape (N, C, H, W)

        Returns
        -------
        The output for each identity training through the model at multiple resolutions if multi-
        scale is enabled
        """
        encoded = [self.encoder(x) for x in inputs]
        if self.architecture == "df":
            decoded = tuple(dec(x) for dec, x in zip(self.decoders, encoded))
        else:
            inters = [torch.concat([(self.inter_both if i == 0 else self.inter_side)(enc),
                                    self.inter_both(enc)],
                                   dim=1)
                      for i, enc in enumerate(encoded)]
            decoded = tuple(self.decoder(x) for x in inters)
        return decoded


__all__ = get_module_objects(__name__)

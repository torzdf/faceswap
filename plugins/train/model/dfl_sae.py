#!/usr/bin/env python3
"""DeepFaceLab SAE Model Based on https://github.com/iperov/DeepFaceLab"""
from __future__ import annotations

import logging

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.nn_blocks import ConvBlockLegacy, ResidualBlockLegacy, UpscaleSubpixel
from lib.utils import FaceswapError, get_module_objects
from plugins.train.train_config import Loss as cfg_loss
from .base import ModelPlugin
from . import dfl_sae_defaults as cfg


logger = logging.getLogger(__name__)


class EncoderDF(nn.Module):
    """The DeepFaceLab SAE-DF Encoder

    Parameters
    ----------
    input_shape
        The (C, H, W) input shape to the model
    encoder_dim
        The number of dimensions per encoder channel
    ae_dims
        The number of dimensions for the latent space
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 encoder_dim: int,
                 ae_dims: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        channels, res = input_shape[:2]
        dims = channels * encoder_dim
        self._lowest_res = res // 16
        self._ae_dims = ae_dims

        self.conv1 = ConvBlockLegacy(channels, dims, 5, stride=2, padding="same")
        self.conv2 = ConvBlockLegacy(dims, dims * 2, 5, stride=2, padding="same")
        self.conv3 = ConvBlockLegacy(dims * 2, dims * 4, 5, stride=2, padding="same")
        self.conv4 = ConvBlockLegacy(dims * 4, dims * 8, 5, stride=2, padding="same")

        self.flatten = nn.Flatten(start_dim=1)
        self.dense1 = nn.Linear(dims * 8 * self._lowest_res * self._lowest_res, self._ae_dims)
        self.dense2 = nn.Linear(self._ae_dims, self._ae_dims * self._lowest_res * self._lowest_res)
        self.upscale = UpscaleSubpixel(self._ae_dims, self._ae_dims)  # TODO check

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DeepFaceLab SAE-DF encoder

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
        x = x.view(x.shape[0], self._ae_dims, self._lowest_res, self._lowest_res)
        return self.upscale(x)


class EncoderLIAE(nn.Module):
    """The DeepFaceLab SAE-LIAE Encoder

    Parameters
    ----------
    input_shape
        The (C, H, W) input shape to the model
    encoder_dim
        The number of dimensions per encoder channel
    """
    def __init__(self,
                 input_shape: tuple[int, int, int],
                 encoder_dim: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        channels = input_shape[0]
        dims = channels * encoder_dim

        self.conv1 = ConvBlockLegacy(channels, dims, 5, stride=2, padding="same")
        self.conv2 = ConvBlockLegacy(dims, dims * 2, 5, stride=2, padding="same")
        self.conv3 = ConvBlockLegacy(dims * 2, dims * 4, 5, stride=2, padding="same")
        self.conv4 = ConvBlockLegacy(dims * 4, dims * 8, 5, stride=2, padding="same")
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DeepFaceLab SAE-LIAE encoder

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
        return self.flatten(x)


class InterLIAE(nn.Module):
    """The DeepFaceLab SAE-LIAE Intermediate layer

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

        self._ae_dims = ae_dims
        self._input_size = input_size
        self.dense1 = nn.Linear(self._input_size[0] * self._input_size[1] * self._input_size[1],
                                self._ae_dims)
        self.dense2 = nn.Linear(self._ae_dims,
                                self._ae_dims * 2 * self._input_size[1] * self._input_size[1])
        self.upscale = UpscaleSubpixel(self._ae_dims * 2, self._ae_dims * 2)  # TODO check

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DeepFaceLab SAE-LIAE Intermediate layer

        Parameters
        ----------
        inputs
            The input to the inter layer

        Returns
        -------
        The output from the inter layer
        """
        x: torch.Tensor = self.dense1(inputs)
        x = self.dense2(x)
        x = x.view(x.shape[0], self._ae_dims * 2, self._input_size[1], self._input_size[1])
        x = self.upscale(x)
        return x


class Decoder(nn.Module):
    """The original DeepFaceLab SAE Decoder Network.

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

        self.upscale1 = UpscaleSubpixel(in_channels, dims * 8)  # TODO size
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2)
        self.res1_1 = ResidualBlockLegacy(dims * 8, dims * 8)
        self.res1_2 = ResidualBlockLegacy(dims * 8, dims * 8)
        if multiscale_count >= 3:
            self.conv_out1 = nn.Conv2d(dims * 8, 3, 5, stride=1, padding=2)

        self.upscale2 = UpscaleSubpixel(dims * 8, dims * 4)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.2)
        self.res2_1 = ResidualBlockLegacy(dims * 4, dims * 4)
        self.res2_2 = ResidualBlockLegacy(dims * 4, dims * 4)
        if multiscale_count >= 3:
            self.conv_out2 = nn.Conv2d(dims * 4, 3, 5, stride=1, padding=2)

        self.upscale3 = UpscaleSubpixel(dims * 4, dims * 2)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.2)
        self.res3_1 = ResidualBlockLegacy(dims * 2, dims * 2)
        self.res3_2 = ResidualBlockLegacy(dims * 2, dims * 2)

        self.conv_out = nn.Conv2d(dims * 2, 3, 5, stride=1, padding=2)

        if learn_mask:
            self.upscale_mask1 = UpscaleSubpixel(in_channels, decoder_dim * 8)
            self.upscale_mask2 = UpscaleSubpixel(decoder_dim * 8, decoder_dim * 4)
            self.upscale_mask3 = UpscaleSubpixel(decoder_dim * 4, decoder_dim * 2)
            self.conv_mask = nn.Conv2d(decoder_dim * 2, 1, 5, stride=1, padding=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass through the DeepFaceLab SAE decoder

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
        x = self.res1_2(self.res1_1(self.leaky1(self.upscale1(inputs))))
        if self._multiscale_count >= 3:
            out.append(torch.sigmoid(self.conv_out1(x)))

        x = self.res2_2(self.res2_1(self.leaky2(self.upscale2(x))))
        if self._multiscale_count >= 2:
            out.append(torch.sigmoid(self.conv_out2(x)))

        x = self.res3_2(self.res3_1(self.leaky3(self.upscale3(x))))
        out.append(torch.sigmoid(self.conv_out(x)))
        if self._learn_mask:
            mask = self.upscale_mask1(inputs)
            mask = self.upscale_mask2(mask)
            mask = self.upscale_mask3(mask)
            out.append(torch.sigmoid(self.conv_mask(mask)))

        return tuple(out)


class DFLSae(ModelPlugin):
    """ DeepFaceLab SAE Faceswap Model.

    Parameters
    ----------
    num_identities
        The number of identities that the model is to be trained on. Default: 2
    """
    def __init__(self, num_identities: int = 2) -> None:
        logger.debug(parse_class_init(locals()))
        if num_identities != 2:
            raise FaceswapError(f"{self.__class__.__name__} only supports 2 identities. Reduce "
                                "the number of identities or choose a different model")
        super().__init__(num_identities, input_size=cfg.input_size())
        self.architecture = cfg.architecture().lower()

        enc_dim = cfg.encoder_dims()
        ae_dims = cfg.autoencoder_dims()
        if ae_dims == 0:
            ae_dims = 512 if self.architecture == "df" else 256
        dec_dim = cfg.decoder_dims()
        ms_count = 3 if cfg.multiscale_decoder() else 1
        dec_in = ae_dims if self.architecture == "df" else ae_dims * 4

        if self.architecture == "df":
            self.encoder = EncoderDF(self.input_shape, enc_dim, ae_dims)
            self.decoders = nn.ModuleList(Decoder(dec_in, cfg_loss.learn_mask(), dec_dim, ms_count)
                                          for _ in range(self.num_identities))
        else:
            int_shape = (enc_dim * 3 * 8, self.input_shape[1] // 16, self.input_shape[1] // 16)
            self.encoder = EncoderLIAE(self.input_shape, enc_dim)
            self.inter_both = InterLIAE(int_shape, ae_dims)
            self.inter_side = InterLIAE(int_shape, ae_dims)
            self.decoder = Decoder(dec_in, cfg_loss.learn_mask(), dec_dim, ms_count)

    def forward(self, inputs: list[torch.Tensor]) -> tuple[tuple[torch.Tensor, ...]]:
        """Forward pass through the DeepFaceLab SAE model

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


if __name__ == "__main__":
    size = 64
    # TODO remove after validation
    i = [torch.rand((1, 3, size, size)), torch.rand((1, 3, size, size))]
    # e = EncoderDF((3, size, size), 42, 512)
    # e = EncoderLIAE((3, size, size), 42)
    # out = e(i[0])
    # print(out.shape)
    p = DFLSae(2)
    print(p)
    # print(dir(list(p.modules())[-1]))
    # print(list(p.modules())[-1].out_channels)
    #
    out_ = p(i)
    print([[k.shape for k in j] for j in out_])

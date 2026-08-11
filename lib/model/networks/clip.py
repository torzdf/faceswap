#!/usr/bin/env python3
""" CLIP: https://github.com/openai/CLIP. This implementation only ports the visual transformer
part of the model.
"""
from __future__ import annotations
import logging
import typing as T
from dataclasses import dataclass
from collections import OrderedDict

import torch
from torch import nn

from lib.model.layers import QuickGELU
from lib.model.weights import GetWeights
from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


TypeModels = T.Literal["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336px",
                       "FaRL-B-16-16", "FaRL-B-16-64"]


@dataclass
class ViTConfig:
    """ Configuration settings for ViT

    Parameters
    ----------
    embed_dim
        Dimensionality of the final shared embedding space
    resolution
        Spatial resolution of the input images
    num_layers
        Number of layers in the visual encoder
    width
        Width of the visual encoder layers
    patch
        Size of the patches to be extracted from the images. Only used for Visual encoder.
    weights
        Available imagenet weights for the config
    """
    embed_dim: int
    resolution: int
    num_layers: int
    width: int
    patch: int
    weights: dict[str, str]


MODEL_CONFIG: dict[TypeModels, ViTConfig] = {
    "ViT-B-16": ViTConfig(embed_dim=512,
                          resolution=224,
                          num_layers=12,
                          width=768,
                          patch=16,
                          weights={"DEFAULT": "clip_vit_b16_imagenet.pth",
                                   "FaRL-B-16-16": "clip_vit_b16_farl_16.pth",
                                   "FaRL-B-16-64": "clip_vit_b16_farl_64.pth"}),
    "ViT-B-32": ViTConfig(embed_dim=512,
                          resolution=224,
                          num_layers=12,
                          width=768,
                          patch=32,
                          weights={"DEFAULT": "clip_vit_b32_imagenet.pth"}),
    "ViT-L-14": ViTConfig(embed_dim=768,
                          resolution=224,
                          num_layers=24,
                          width=1024,
                          patch=14,
                          weights={"DEFAULT": "clip_vit_l14_imagenet.pth"}),
    "ViT-L-14-336px": ViTConfig(embed_dim=768,
                                resolution=336,
                                num_layers=24,
                                width=1024,
                                patch=14,
                                weights={"DEFAULT": "clip_vit_l14_336_imagenet.pth"})}


# ################## #
# VISUAL TRANSFORMER #
# ################## #
class ResidualAttentionBlock(nn.Module):
    """ Residual Attention Block for Visual Transformer

    Parameters
    ----------
    embed_dim
        embedding dimension for MultiHeadAttention
    num_heads
        Number of heads for MultiHeadAttention
    attn_mask
        The attention mask. Default: ``None``
    """
    def __init__(self, embed_dim: int, num_heads: int, attn_mask: torch.Tensor | None = None
                 ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(embed_dim * 4, embed_dim))
        ]))
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn_mask = attn_mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward through the Residual Attention Block

        Parameters
        ----------
        inputs
            The input tensor to the layer

        Returns
        -------
        The output tensor from the layer
        """
        x = self.ln_1(inputs)
        attn = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        x = inputs + attn
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Sequential):
    """ A class representing a Transformer model with attention mechanism and residual connections.

    Parameters
    ----------
    width
        The dimension of the input and output vectors.
    num_layers
        The number of layers in the Transformer.
    heads
        The number of attention heads.
    attn_mask
        The attention mask. Default: ``None``
    """
    def __init__(self,
                 width: int,
                 num_layers: int,
                 heads: int,
                 attn_mask: torch.Tensor | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        for _ in range(num_layers):
            self.append(ResidualAttentionBlock(width, heads, attn_mask))


class VisualTransformer(nn.Module):
    """ A class representing a Visual Transformer model for image classification tasks.

    Parameters
    ----------
    input_resolution
        The input resolution of the images.
    patch_size
        The size of the patches to be extracted from the images.
    width
        The dimension of the input and output vectors.
    num_layers
        The number of layers in the Transformer.
    heads
        The number of attention heads.
    output_dim
        The dimension of the output vector.
    """
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 num_layers: int,
                 heads: int,
                 output_dim: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, num_layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward through the Visual Transformer

        Parameters
        ----------
        inputs
            The input tensor to the layer

        Returns
        -------
        The output tensor from the layer
        """
        x: torch.Tensor = self.conv1(inputs)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.expand(x.shape[0], 1, -1), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


def _get_vision_net(resolution: int | None, config: ViTConfig) -> VisualTransformer:
    """ Obtain a Visual Transformer model for the given configuration """
    res = config.resolution if resolution is None else resolution
    return VisualTransformer(input_resolution=res,
                             width=config.width,
                             num_layers=config.num_layers,
                             output_dim=config.embed_dim,
                             heads=config.width // 64,
                             patch_size=config.patch)


def _prepare_model(weights: T.Literal["DEFAULT", "FaRL-B-16-16", "FaRL-B-16-64"] | None,
                   conf: ViTConfig,
                   input_size: int | None) -> VisualTransformer:
    """ Prepare a model with the given configuration and weights """
    if weights is not None and weights not in conf.weights:
        logger.warning("Invalid weights type: '%s'. Falling back to 'DEFAULT'", weights)
        weights = "DEFAULT"

    retval = _get_vision_net(input_size, conf)
    if weights is not None:
        weights_file = GetWeights(conf.weights[weights]).model_path
        assert isinstance(weights_file, str)
        state_dict: OrderedDict[str, torch.Tensor] = torch.load(weights_file, map_location="cpu")

        strict = True
        if input_size != 224:
            logger.warning("[ClipV] Positional Embedding weights not loaded as input size != "
                           "224px. All other layers loaded.")
            del state_dict["positional_embedding"]
            strict = False
        retval.load_state_dict(state_dict, strict=strict)
    return retval


def vit_b_16(weights: T.Literal["DEFAULT", "FaRL-B-16-16", "FaRL-B-16-64"] | None = None,
             input_size: int | None = None) -> VisualTransformer:
    """ Obtain a B16 Visual Transformer Model

    Parameters
    ----------
    input_size
        The input size to the Visual Transformer. Default: ``None`` (model default)
    weights
        "DEFAULT" to load imagenet trained weights or "FaRL-B-16-16"/"FaRL-B-16-64" for FaRL
        weights
    """
    return _prepare_model(weights, MODEL_CONFIG["ViT-B-16"], input_size)


def vit_b_32(weights: T.Literal["DEFAULT"] | None = None,
             input_size: int | None = None) -> VisualTransformer:
    """ Obtain a B32 Visual Transformer Model

    Parameters
    ----------
    input_size
        The input size to the Visual Transformer. Default: ``None`` (model default)
    weights
        "DEFAULT" to load imagenet trained weights
    """
    return _prepare_model(weights, MODEL_CONFIG["ViT-B-32"], input_size)


def vit_l_14(weights: T.Literal["DEFAULT"] | None = None,
             input_size: int | None = None) -> VisualTransformer:
    """ Obtain a L14 Visual Transformer Model

    Parameters
    ----------
    input_size
        The input size to the Visual Transformer. Default: ``None`` (model default)
    weights
        "DEFAULT" to load imagenet trained weights
    """
    return _prepare_model(weights, MODEL_CONFIG["ViT-L-14"], input_size)


def vit_l_14_336px(weights: T.Literal["DEFAULT"] | None = None,
                   input_size: int | None = None) -> VisualTransformer:
    """ Obtain a L14 336px Visual Transformer Model

    Parameters
    ----------
    input_size
        The input size to the Visual Transformer. Default: ``None`` (model default)
    weights
        "DEFAULT" to load imagenet trained weights
    """
    return _prepare_model(weights, MODEL_CONFIG["ViT-L-14-336px"], input_size)


__all__ = get_module_objects(__name__)

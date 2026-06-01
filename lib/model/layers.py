#!/usr/bin/env python3
"""Custom Layers for faceswap.py."""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class SamePad2d(nn.Module):
    """Asymmetric padding to replicate Keras' padding='same' for backwards compatibility

    Parameters
    ----------
    kernel_size
        The size of the kernel for the following convolution
    stride
        The size of the stride for the following convolution
    """
    def __init__(self, kernel_size: int, stride: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.kernel = kernel_size
        self.stride = stride

    def __repr__(self) -> str:
        """Better info for debug output"""
        return f"{self.__class__.__name__}(kernel_size={self.kernel}, stride={self.stride})"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric padding to the input tensor

        Parameters
        ----------
        inputs
            The tensor to be padded

        Returns
        -------
        The padded tensor
        """
        height, width = inputs.shape[-2:]
        pad_h = int(max((np.ceil(height / self.stride) - 1) * self.stride + self.kernel - height,
                        0))
        pad_w = int(max((np.ceil(width / self.stride) - 1) * self.stride + self.kernel - width,
                        0))
        return F.pad(inputs, (pad_w // 2, pad_w - pad_w // 2,
                              pad_h // 2, pad_h - pad_h // 2))


__all__ = get_module_objects(__name__)

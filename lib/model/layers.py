#!/usr/bin/env python3
"""Custom Layers for faceswap.py."""
from __future__ import annotations

import logging
import math
import typing as T
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch import jit

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class SamePad2d(nn.Module):
    """Asymmetric padding to replicate Keras' padding='same' for backwards compatibility. This
    should not be used to new models. It exists purely to enable bit-accurate porting of Tensorflow
    built models into Torch

    Parameters
    ----------
    kernel_size
        The size of the kernel for the following convolution
    stride
        The size of the stride for the following convolution
    padding_mode
        The type of padding to apply. Default: `constant`
    """
    def __init__(self,
                 kernel_size: int,
                 stride: int,
                 mode: T.Literal["constant", "reflect"] = "constant") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.kernel = kernel_size
        self.stride = stride
        self.mode = mode

    def __repr__(self) -> str:
        """Better info for debug output"""
        return (f"{self.__class__.__name__}(kernel_size={self.kernel}, "
                f"stride={self.stride}, mode={self.mode})")

    def _pad(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform asymmetric padding to the input tensor

        Parameters
        ----------
        inputs
            The input to the padding layer

        Returns
        -------
        The padded input tensor
        """
        height, width = inputs.shape[-2:]
        pad_h = max((math.ceil(height / self.stride) - 1) * self.stride + self.kernel - height, 0)
        pad_w = max((math.ceil(width / self.stride) - 1) * self.stride + self.kernel - width, 0)
        return F.pad(inputs,
                     (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2),
                     mode=self.mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the legacy padding layer

        Parameters
        ----------
        inputs
            The tensor to be padded

        Returns
        -------
        The padded tensor
        """
        if jit.is_tracing():  # Hide nasty constant warnings when JIT tracing
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="Converting a tensor to a Python",
                                        category=jit.TracerWarning)
                return self._pad(inputs)
        return self._pad(inputs)


__all__ = get_module_objects(__name__)

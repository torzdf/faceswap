#!/usr/bin/env python3
"""Base class for Models plugins ALL Models should at least inherit from this class."""

import abc
import logging

from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects


logger = logging.getLogger(__name__)


class ModelPlugin(nn.Module, abc.ABC):
    """Parent class for all models to inherit from

    Parameters
    ----------
    num_identities
        The number of identities the model is being trained on
    input_size
        The pixel input size to the model
    version
        The plugin version. Versions less than 1.0 means that the model was created in Keras.
        Versions 1.0 and above are created in Torch
    is_rgb
        ``True`` for rgb. ``False`` for bgr. Default: ``False`` (bgr)
    """
    def __init__(self,
                 num_identities: int,
                 input_size: int | None = None,
                 version: float | None = None,
                 is_rgb: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        assert input_size is not None, "input_size should be provided by plugin"
        assert version is not None, "version should be provided by plugin"

        self.num_identities = num_identities
        self.version = version
        self.input_shape = (3, input_size, input_size)
        self.is_rgb = is_rgb
        super().__init__()

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k}={v}" for k, v in {"num_identities": self.num_identities,
                                                   "version": self.version,
                                                   "input_size": self.input_shape[1],
                                                   "is_rgb": self.is_rgb}.items())
        return f"{self.__class__.__name__}({params})"


__all__ = get_module_objects(__name__)

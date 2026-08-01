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
        The pixel input size to the model. Default: 0 (invalid)
    is_rgb
        ``True`` for rgb. ``False`` for bgr. Default: ``False`` (bgr)
    is_legacy
        ``True`` if the model was originally created in Keras. Default ``False``
    """
    def __init__(self,
                 num_identities: int,
                 input_size: int = 0,
                 is_rgb: bool = False,
                 is_legacy: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        assert input_size > 0
        self.num_identities = num_identities
        self.input_shape = (3, input_size, input_size)
        self.is_rgb = is_rgb
        self.is_legacy = is_legacy
        super().__init__()

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k}={v}" for k, v in {"num_identities": self.num_identities,
                                                   "input_size": self.input_shape[1],
                                                   "is_rgb": self.is_rgb,
                                                   "is_legacy": self.is_legacy}.items())
        return f"{self.__class__.__name__}({params})"


__all__ = get_module_objects(__name__)

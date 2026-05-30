#!/usr/bin/env python3
"""Base class for Models plugins ALL Models should at least inherit from this class."""

import abc
from torch import nn


class ModelPlugin(nn.Module, abc.ABC):
    """Parent class for all models to inherit from

    Parameters
    ----------
    input_size
        The pixel input size to the model
    """
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        super().__init__()

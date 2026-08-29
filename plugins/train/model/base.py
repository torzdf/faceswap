#!/usr/bin/env python3
""" Base class for Models plugins

This module defines the abstract ``ModelPlugin`` base class from which every model plugin must
inherit, standardising the shared interface and configuration that all models supply to the
training workflow so a reader can understand what a plugin is without inspecting each
implementation
"""
import abc
import logging
import typing as T

from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from plugins.train.train_config import load_config

logger = logging.getLogger(__name__)


class ModelPlugin(nn.Module, abc.ABC):
    """ Abstract base class for all face-swap model plugins

    Defines the common interface and configuration shared by every model plugin so each network is
    trained and applied consistently. As an abstract ``nn.Module`` it cannot be instantiated
    directly; concrete subclasses must supply their architecture together with the arguments below

    The module also implements a simple versioning scheme: models created with ``version`` below
    1.0 were built using the legacy Keras convention, while versions at or above 1.0 use the Torch-
    native implementation. Callers can rely on this flag to select compatible layers without
    inspecting each model individually

    Parameters
    ----------
    num_identities
        The number of distinct identities (faces) the model is being trained against
    input_size
        The pixel size, in pixels, of a single face image fed into the network. This must be
        provided by the subclass; there is no sensible default and passing ``None`` raises an
        assertion error
    version
        The plugin version number used to distinguish legacy Keras models from Torch-native ones.
        This must also be provided by the subclass
    is_rgb, optional
        Whether the network expects ``rgb`` or ``bgr`` input. Default: ``False`` (bgr)

    Notes
    -----
    Construction records only this common setup; no layers are defined here and all model-specific
    behaviour lives in the subclass.
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
        """ The number of distinct identities (faces) the model is being trained against """
        self.version = version
        """ The plugin version number of the plugin """
        self.input_shape = (3, input_size, input_size)
        """ The (C, H, W) input shape to the model """
        self.is_rgb = is_rgb
        """ ``True`` if the model expects RGB input """
        self.name = self.__class__.__module__.rsplit(".")[-1]
        """ The internal name of the model """
        super().__init__()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = ", ".join(f"{k}={v!r}" for k, v in {"num_identities": self.num_identities,
                                                     "version": self.version,
                                                     "input_size": self.input_shape[1],
                                                     "is_rgb": self.is_rgb}.items())
        return f"{self.__class__.__name__}({params})"

    def _config_layers(self, key: T.Literal["freeze_layers", "load_layers"]) -> list[str]:
        """ Return the current value of the plugin's load/freeze layers setting

        Parameters
        ----------
        key
            The layer setting to look up in config

        Returns
        -------
        The currently configured setting for the given key or empty list if key doesn't exist
        """
        conf = load_config().sections[f"model.{self.name}"].options.get(key)
        logger.debug("%s Got config item for '%s': %s", self.__class__.__name__, key, conf)
        retval = [] if conf is None else conf()
        if conf is None and "encoder" in list(dict(self.named_children())):
            logger.debug("%s Defaulting to `encoder` for '%s' without a config option",
                         self.__class__.__name__, key)
            retval = ["encoder"]
        return retval

    @property
    def freeze_layers(self) -> list[str]:
        """ The currently configured layers to keep frozen during training """
        return self._config_layers("freeze_layers")

    @property
    def load_layers(self) -> list[str]:
        """ The currently configured layers to load weights into during training """
        return self._config_layers("load_layers")


__all__ = get_module_objects(__name__)

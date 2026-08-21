#! /usr/env/bin/python3
"""Handlers for creating training and inference objects from Faceswap model plugins"""
from __future__ import annotations

import logging
import os
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.initializers import icnr, ConvolutionAware
from lib.utils import get_module_objects

from plugins.plugin_loader import PluginLoader
from plugins.train import train_config as mod_cfg

from .model_info import Info
from .saving import ModelIO
from .train_state import State

if T.TYPE_CHECKING:
    from plugins.train.model.base import ModelPlugin
    from .model_info import Layer


logger = logging.getLogger(__name__)


class TrainConfigure:
    """Configures a Faceswap model for training based on user provided values

    Parameters
    ----------
    model
        The faceswap model object to be configured
    icnr_init
        ``True`` to initialize convolutions prior to up-scales with ICNR
    conv_aware_init
        ``True`` to apply conv_aware_init to all convolutions
    reflect_padding
        ``True`` to apply reflect padding to convolutions
    """
    def __init__(self,
                 model: FaceswapModel,
                 icnr_init: bool,
                 conv_aware_init: bool,
                 reflect_padding: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = model
        self._init = {"icnr": icnr_init, "conv_aware": conv_aware_init}
        self._reflect_padding = reflect_padding

    def _get_prev_conv(self, layer: Layer, collected: list[Layer] | None = None) -> list[Layer]:
        """Recurse backwards through the model info to get the next Convolution layer that exists
        prior to the given layer

        Parameters
        ----------
        layer
            The layer to work backwards from
        collected
            List of found convolutions, for recursion

        Returns
        -------
        The next convolutions prior to the given layer (multiple if path splits prior to a conv
        being found)
        """
        collected = [] if collected is None else collected
        if layer.type == "Conv2d":
            return collected + [layer]
        for lyr in layer.input_layers:
            return self._get_prev_conv(self._model.info.structure[lyr], collected)
        return collected

    def _apply_initializers(self, model: ModelPlugin) -> None:
        """Apply the requested initializers to the relevant convolutions

        Parameters
        ----------
        model
            The Faceswap model to update the initializers for
        """
        if not any(self._init.values()):
            logger.debug("[TrainConfigure] No custom initializers to apply")
            return
        # TODO prevent running on ImageNet weights load
        conv_aware = ConvolutionAware()
        icnr_conv = [x.name for v in self._model.info.structure.values()
                     if v.type == "PixelShuffle"
                     for x in self._get_prev_conv(v)] if self._init["icnr"] else []
        for k, v in model.named_modules():
            if k in icnr_conv and isinstance(v, nn.Conv2d):
                logger.debug("[TrainConfigure] Applying ICNR Initialization: '%s' (%s)",
                             k, v.weight.shape)
                icnr(v.weight)
                if v.bias is not None:
                    nn.init.zeros_(v.bias)
            elif self._init["conv_aware"] and isinstance(v, nn.Conv2d):
                logger.info("[TrainConfigure] Applying ConvAware Init '%s' %s...",
                            k, tuple(v.weight.shape))
                conv_aware(v.weight)
                if v.bias is not None:
                    nn.init.zeros_(v.bias)

    def _apply_reflect_padding(self, model: ModelPlugin) -> None:
        """Apply reflect padding on qualifying convolution layers

        Parameters
        ----------
        model
            The Faceswap model to apply reflect padding to
        """
        if not self._reflect_padding:
            logger.debug("[TrainConfigure] No reflect padding to apply")
            return
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            pad = module.padding
            stride = module.stride
            kern = module.kernel_size
            if all(p == 0 for p in (pad if isinstance(pad, tuple) else (pad, pad))):
                logger.debug("[TrainConfigure] Skip conv '%s' with zero padding: %s",
                             name, pad)
                continue
            if module.padding_mode != "zeros":
                logger.debug("[TrainConfigure] Skip conv '%s' with non-zero padding: %s",
                             name, repr(module.padding_mode))
                continue
            if all(k == 1 for k in (kern if isinstance(kern, tuple) else (kern, kern))):
                logger.debug("[TrainConfigure] Skip conv '%s' with kernel size == 1", name)
                continue
            if any(s > 1 for s in (stride if isinstance(stride, tuple) else (stride, stride))):
                logger.debug("[TrainConfigure] Skip conv '%s' with stride > 1: %s",
                             name, stride)
                continue
            logger.debug("[TrainConfigure] Reflect pad conv '%s'. padding: %s, kernel: %s, "
                         "stride: %s, original mode: %s",
                         name, pad, module.kernel_size, module.stride, module.padding_mode)
            module.padding_mode = "reflect"

    def configure(self) -> None:
        """Configure the given faceswap model with the user provided settings """
        self._apply_initializers(self._model.plugin)
        self._apply_reflect_padding(self._model.plugin)
        # TODO MSG
        logger.debug("[TrainConfigure] Configured model")


class FaceswapModel:
    """ A container for a loaded or newly created Faceswap model plugin

    This class manages the lifecycle of faceswap models, including loading saved checkpoints,
    creating new instances, handling model configuration, and handling state management. It serves
    as the interface between the training and inference systems and the underlying neural network
    plugins.

    Parameters
    ----------
    name
        The identifier/name of the model to load (plugin name)
    model_dir
        Directory path where model weights are stored/loaded from
    num_identities
        Number of identity mappings in this model
    load_optimizer
        ``True`` to attempt loading optimizer state from disk. Only works if a previous checkpoint
        exists with saved optimizer state. Default: ``False``
    config_file
        Optional path to additional configuration file for this model

    Attributes
    ----------
    name
        The normalized plugin name (with dashes replaced by underscores)
    io
        ModelIO handler for loading/saving operations on this model's weights and state
    state
        State object tracking iterations, session progress, training metadata and model
        configuration
    plugin
        The underlying torch.nn.Module that performs the actual face swapping
    """
    def __init__(self,
                 name: str,
                 model_dir: str,
                 num_identities: int,
                 load_extra_state: bool = False,
                 config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))

        mod_cfg.load_config(config_file=config_file)  # Set global config

        self._load_extra_state = load_extra_state
        self._conf_file = config_file
        self._log_name = f"[{self.__class__.__name__}.{name}]"
        self._info: Info | None = None

        self.name = name.replace("-", "_")
        """ The plugin name of the model to load """
        self.state = State(PluginLoader.get_model_path(name, module=True))
        """ The training and configuration state of the model """
        self.io = ModelIO(name, model_dir)
        """ Handles loading and saving operations for the model and associated files """
        self.plugin, self._extra_state = self._load_plugin(num_identities)
        """ The loaded Faceswap plugin """

        self._configure()

    def __repr__(self) -> str:
        """ String representation of the FaceswapModel for debugging and logging """
        params = {"name": repr(self.name.replace("-", "_")),
                  "model_dir": repr(os.path.dirname(self.io.checkpoint_path)),
                  "num_identities": repr(self.plugin.num_identities),
                  "load_extra_state": repr(self._load_extra_state),
                  "config_file": repr(self._conf_file)}
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @property
    def info(self) -> Info:
        """ Information about the currently loaded Faceswap Model. This provides metadata about
        layers, shapes, and model structure. """
        if self._info is None:
            self._info = Info(self.plugin)
        return self._info

    def _create_new(self, num_identities: int) -> ModelPlugin:
        """ Create a new Faceswap model plugin from scratch

        Used when no saved checkpoint exists. Creates a fresh instance with the
        specified number of identities for the current global configuration.

        Parameters
        ----------
        num_identities
            The number of identity mappings for this model

        Returns
        -------
        A new ModelPlugin instance
        """
        logger.debug("[TrainHandler] No state_dict to load. Creating new model")
        plugin = PluginLoader.get_model(self.name)(num_identities=num_identities)
        self.state.set_plugin_version(plugin.version)
        return plugin

    def _load_existing(self, num_identities: int) -> tuple[ModelPlugin, dict[str, T.Any]]:
        """ Load a saved Faceswap model from disk

        Loads both the model weights and state dictionary. If no state is found in
        the checkpoint, falls back to creating a new model with default parameters.

        Parameters
        ----------
        num_identities
            The number of identities expected by this model

        Returns
        -------
        A tuple containing (model_plugin, optimizer_state) where optimizer_state may
        be ``None`` if not present in the checkpoint or this is a fresh model

        Notes
        -----
        Only loads state_dict entries for "model", "state", and "version" from disk.
        Extra states (eg optimizer weights) are returned separately to allow selective loading.
        """
        state_dict = self.io.load()
        if "state" not in state_dict:
            logger.warning("%s No state found in saved config. Loading from model defaults.",
                           self._log_name)
            return self._create_new(num_identities), {}

        logger.info("%s Loading plugin from saved config", self._log_name)
        self.state.load_state_dict(T.cast(dict[str, T.Any], state_dict["state"]))
        plugin = PluginLoader.get_model(self.name)(num_identities=num_identities,
                                                   version=self.state.plugin_version)
        plugin.load_state_dict(state_dict["model"])
        self.state.load_state_dict(state_dict["state"])

        extra_state = {k: v for k, v in state_dict.items()
                       if k not in ("state", "model", "version")} if self._load_extra_state else {}

        return plugin, extra_state

    def _load_plugin(self, num_identities: int) -> tuple[ModelPlugin, dict[str, T.Any]]:
        """ Determine whether to create new model or load existing one

        Checks if a checkpoint file exists at the model path. Creates fresh instance
        on first run, loads saved weights on subsequent runs.

        Parameters
        ----------
        num_identities
            Required number of identity mappings for this model

        Returns
        -------
        A tuple containing (model_plugin, optimizer_state) where `optimizer_state` is ``None``
        unless previously saved and `load_optimizer` was ``True``

        Notes
        -----
        Optimizer state is only cached temporarily during training initialization to avoid loading
        from disk twice.
        """
        if not self.io.file_exists:
            return self._create_new(num_identities), {}
        return self._load_existing(num_identities)

    def _configure(self) -> None:
        """ Configure the model for training based on initialization settings

        Applies ICNR initialization, conv_aware_init, and reflect_padding according
        to state configuration. Skips these operations if weights were loaded from
        disk (to preserve trained weights). Calls TrainConfigure.configure() to apply
        all settings in one step.
        """
        icnr = False if self.io.file_exists else self.state.config.get("icnr_init", False)
        conv = False if self.io.file_exists else self.state.config.get("conv_aware_init", False)
        reflect = self.state.config.get("reflect_padding", False)
        TrainConfigure(self,
                       icnr_init=icnr,
                       conv_aware_init=conv,
                       reflect_padding=reflect).configure()

    def state_dict(self) -> dict[str, T.Any]:
        """ Get the model's complete state dictionary

        Returns a dictionary containing the plugin weights, training state, and version.
        Does NOT include extra_state. Use pop_extra_state() for that.

        Returns
        -------
        A dict with keys: "model", "state", "version" representing all trainable
        parameters and configuration of this Faceswap model
        """
        return {"version": 1.0,
                "model": self.plugin.state_dict(),
                "state": self.state.state_dict()}

    def pop_extra_state(self, key: str) -> dict[str, T.Any] | None:
        logger.debug("%s Popping extra_state: '%s'", self._log_name, key)
        return self._extra_state.pop(key, None)

    def clear_extra_state(self) -> None:
        if not self._extra_state:
            logger.debug("%s extra_state is already empty.", self._log_name)
            return
        logger.debug("%s Clearing from extra_state: %s",
                     self._log_name, list(self._extra_state))
        self._extra_state.clear()

    def to(self, device: torch.Device) -> None:
        """ Move the model plugin to a different computational device

        Transfers all parameters and buffers of the underlying neural network module
        from CPU/GPU to the specified device. Does not affect state dictionaries or
        configuration objects.

        Parameters
        ----------
        device
            The target torch.Device (cuda:X, cpu,mps:0, etc.) where model should live
        """
        logger.debug("%s Model to: %s", self._log_name, device)
        self.plugin.to(device)


__all__ = get_module_objects(__name__)

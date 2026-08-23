#! /usr/env/bin/python3
""" Handles Faceswap model loading, initialization, and configuration including weight initializers

This module provides the TrainConfigure utility class for applying custom weight initializations
(ICNR, ConvolutionAware) and padding modifications (reflect mode) to model convolutions before
training begins. It also contains the FaceswapModel container class which manages plugin loading,
state persistence via ModelIO, and configuration of neural network components during model setup
"""
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
    """ Utility class for configuring model initialization and padding settings

    TrainConfigure is a temporary configuration helper used only during model setup to apply
    custom weight initializations (ICNR or ConvolutionAware) and modify convolution padding modes
    from 'zeros' to 'reflect'. All effects are applied in-place to the model instance before
    training/inference begins and are dictated by the user config settings when initially creating
    the model

    The class performs three main operations:
        1. ICNR initialization : Applied to convolutions preceding PixelShuffle layers, improving
           super-resolution output quality by correlating kernel values with noise magnitude
        2. ConvolutionAware init : Applies specialized weight initialization for encoder-decoder
           architectures common in faceswap models, setting weights based on layer position
        3. Reflect padding: Changes convolution padding from 'zeros' to 'reflect' mode to prevent
           boundary artifacts that can occur with zero-padding at image edges

    Parameters
    ----------
    model
        Reference to the parent FaceswapModel containing the plugin being configured. Used for
        accessing layer structure information when determining ICNR target layers as well as
        holding the model's nn.Module for amending
    icnr_init
        Whether to apply ICNR initialization to convolutions preceding PixelShuffle layers. This
        is enabled during fresh model creation but disabled when resuming/running inference
    conv_aware_init
        Whether to apply ConvolutionAware initialization to all Conv2d layers. Similar to ICNR,
        this is only applied on fresh models and skipped when restoring from saved checkpoints
    reflect_padding
        Whether to change convolution padding mode from 'zeros' to 'reflect'. This prevents
        boundary artifacts at image edges during inference or training

    Notes
    -----
    This utility is instantiated during FaceswapModel._configure() and only affects the plugin
    passed to it. The configuration is saved within the State property of the FaceswapModel

    ICNR initialization specifically targets convolutions that feed into PixelShuffle layers, which
    are common in super-resolution architectures used for face swapping quality enhancement
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
        """ Recursively find all Conv2d layers preceding a given layer in the model structure

        Parameters
        ----------
        layer
            Starting point for traversal. Normally PixelShuffle layer whose convs need ICNR
        collected
            Accumulator list from recursive calls. Default: ``None`` (create new empty list)

        Returns
        -------
        List of Conv2d Layer objects found during traversal from root down to the target layer.
        """
        collected = [] if collected is None else collected
        if layer.type == "Conv2d":
            return collected + [layer]
        for lyr in layer.input_layers:
            return self._get_prev_conv(self._model.info.structure[lyr], collected)
        return collected

    def _apply_initializers(self, model: ModelPlugin) -> None:
        """ Apply custom weight initialization to target convolutional layers

        Parameters
        ----------
        model
            The plugin whose convolutional layers should receive custom initialization.
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
        """ Change convolution padding mode from 'zeros' to 'reflect' for eligible layers

        Parameters
        ----------
        model
            The plugin whose convolutional layers should have padding modes updated to 'reflect'
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
        """ Apply all configured initialization and padding modifications to the model

        Executes in this sequence:
            1. Applies ICNR or ConvolutionAware weight init if enabled
            2. Changes padding modes for eligible convolutions

        Both operations modify the plugin instance passed during __init__ and persist changes to
        memory. These are in-memory modifications only affecting this model instance

        Notes
        -----
        This method is called from FaceswapModel._configure() after plugin instantiation but before
        training/inference begins. Once complete, the model has proper weight initializations and
        padding configurations applied for optimal training convergence and inference quality
        """
        self._apply_initializers(self._model.plugin)
        self._apply_reflect_padding(self._model.plugin)
        # TODO MSG
        logger.debug("[TrainConfigure] Configured model")


class FaceswapModel:
    """ Container class for loading, managing, and configuring neural network models

    FaceswapModel wraps a ModelPlugin instance (neural network architecture) along with associated
    state management objects for persistence (State, ModelIO). It handles model creation from
    scratch, loading from saved checkpoints, plugin configuration (weight initialization, padding),
    and device placement during runtime. The class supports multiple identity models loaded
    simultaneously via the num_identities parameter which determines how many face encodings are
    processed per forward pass

    Loading Flow:
    -----------
    1. Loads config file if provided, instantiates plugin from PluginLoader
    2. Applies weight initializers (ICNR/ConvAware) and padding modifications via TrainConfigure
    3. Manages checkpoint serialization/deserialization

    State Management:
    -------------
    The State object tracks plugin version, configuration parameters (icnr_init, conv_aware_init),
    and model architecture details. ModelIO handles file I/O for loading/saving checkpoints from
    disk. Extra state (non-standard attributes) can be persisted and managed separately via
    pop_extra_state().

    Parameters
    ----------
    name
        Model identifier string used to look up the plugin in PluginLoader.get_model(). Dashes are
        converted to underscores for internal attribute access consistency throughout the class
    model_dir
        Directory path containing saved model checkpoint files (model.pth, model.ckpt,
        config.json). Used by ModelIO to locate and load existing checkpoints or create new ones
        if needed
    num_identities
        Number of face encodings processed simultaneously per forward pass. Larger values enable
        multi-face processing but increase computational cost linearly with the number provided
    load_extra_state, optional
        Whether to persist non-standard attributes in state_dict beyond version/model/state keys
        when loading weights. When ``True``, extra information in the state_dict are kept loaded in
        the object.When ``False``, any extra state information is discarded. Default: ``False``
    config_file, optional
        Path to custom configuration file overriding defaults from mod_cfg.train_config().
        If ``None``, uses built-in config

    Notes
    -----
    Model loading behavior depends on whether checkpoint file exists at model_dir:
        - File exists : Loads existing weights and state including version compatibility handling
        - No file : Creates fresh plugin with random initialization

    Extra state management supports persistence of custom attributes beyond standard PyTorch
    state_dict keys. This is useful for storing metadata or non-tensor parameters that are part of
    the training loop

    The info property uses lazy loading pattern - first access triggers Info(self.plugin) creation
    which caches result
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
        """ Model identifier with dashes replaced by underscores for internal consistency """

        self.state = State(PluginLoader.get_model_path(name, module=True))
        """ State management object tracking plugin version and configuration parameters """
        self.io = ModelIO(name, model_dir)
        """ File I/O handler for loading/saving checkpoints from model_dir directory """

        self.plugin: ModelPlugin
        """ The instantiated neural network module ready for training or inference after
        configuration completes """

        self.plugin, self._extra_state = self._load_plugin(num_identities)

        self._configure()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = {"name": repr(self.name.replace("-", "_")),
                  "model_dir": repr(os.path.dirname(self.io.checkpoint_path)),
                  "num_identities": repr(self.plugin.num_identities),
                  "load_extra_state": repr(self._load_extra_state),
                  "config_file": repr(self._conf_file)}
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @property
    def info(self) -> Info:
        """ Model information object containing architecture details """
        if self._info is None:
            self._info = Info(self.plugin)
        return self._info

    def _create_new(self, num_identities: int) -> ModelPlugin:
        """ Instantiate a fresh plugin without loading any saved state

        Parameters
        ----------
        num_identities : int
            Number of face encodings that this model is configured to support

        Returns
        -------
        ModelPlugin
            Fresh plugin instance with random weights ready for training
        """
        logger.debug("[FaceswapModel] No state_dict to load. Creating new model")
        plugin = PluginLoader.get_model(self.name)(num_identities=num_identities)
        self.state.set_plugin_version(plugin.version)
        return plugin

    def _load_existing(self, num_identities: int) -> tuple[ModelPlugin, dict[str, T.Any]]:
        """ Load existing plugin from checkpoint file including model weights and state
        dictionaries

        Parameters
        ----------
        num_identities
            Number of identities the model supports. must match what was used when model created

        Returns
        -------
        plugin
            ModelPlugin instance loaded with weights from checkpoint file
        extra_state
            Dictionary of items (training data loaded from checkpoint) if load_extra_state=True
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
        """ Determine whether to create new or load existing plugin based on file existence check

        Parameters
        ----------
        num_identities : int
            Number of identities model supports

        Returns
        -------
        plugin
            ModelPlugin instance loaded with weights from checkpoint file
        extra_state
            Dictionary of items (training data loaded from checkpoint) if load_extra_state=True
        """
        if not self.io.file_exists:
            return self._create_new(num_identities), {}
        return self._load_existing(num_identities)

    def _configure(self) -> None:
        """ Apply weight initializations and padding modifications after plugin instantiation """
        icnr_ = False if self.io.file_exists else self.state.config.get("icnr_init", False)
        conv = False if self.io.file_exists else self.state.config.get("conv_aware_init", False)
        reflect = self.state.config.get("reflect_padding", False)
        TrainConfigure(self,
                       icnr_init=icnr_,
                       conv_aware_init=conv,
                       reflect_padding=reflect).configure()

    def state_dict(self) -> dict[str, T.Any]:
        """ Serialize model weights, plugin version, and state metadata for checkpoint saving

        Returns dictionary containing three keys:
            - "version" : Always 1.0 to indicate compatibility with current loader code
            - "model" : PyTorch model weights from self.plugin.state_dict()
            - "state" : State object data including plugin_version and configuration parameters

        This format is saved to checkpoint files via ModelIO.save() for later restoration during
        training or inference sessions

        Returns
        -------
        Serialized state dictionary ready for saving with model weights and version metadata
        """
        return {"version": 1.0,
                "model": self.plugin.state_dict(),
                "state": self.state.state_dict()}

    def pop_extra_state(self, key: str) -> dict[str, T.Any] | None:
        """ Remove and return a specific extra state item by key if it exists in extra_state
        dictionary

        Called by the training loop to retrieve and remove custom attributes that were persisted
        for training purposes

        Parameters
        ----------
        key
            Key of the extra state item to remove from _extra_state dictionary.

        Returns
        -------
        The removed value if key exists in extra_state, otherwise returns ``None``

        Notes
        -----
        This method only affects non-standard attributes stored in self._extra_state - standard
        model weights and state are managed through regular PyTorch state_dict/load_state_dict
        mechanisms.
        """
        logger.debug("%s Popping extra_state: '%s'", self._log_name, key)
        return self._extra_state.pop(key, None)

    def clear_extra_state(self) -> None:
        """ Clear all non-standard attributes from _extra_state dictionary to free memory or reset
        configuration

        Called when needing to remove all custom metadata that was persisted during training
        session. This is called by the training loop just before running the first iteration
        """
        if not self._extra_state:
            logger.debug("%s extra_state is already empty.", self._log_name)
            return
        logger.debug("%s Clearing from extra_state: %s",
                     self._log_name, list(self._extra_state))
        self._extra_state.clear()

    def to(self, device: torch.Device) -> None:
        """ Move the plugin module (and its weights/parameters) to specified device for runtime
        execution

        Called during training loop initialization or inference setup to ensure model runs on
        correct computational device (CPU/GPU). Updates self.plugin's .to() method which moves all
        parameters and buffers accordingly

        Parameters
        ----------
        device
            The device to place the model

        Notes
        -----
        This method delegates to self.plugin.to(device) - the actual PyTorch mechanism for moving
        model to different devices at runtime without modifying weights themselves (just parameter
        storage location)
        """
        logger.debug("%s Model to: %s", self._log_name, device)
        self.plugin.to(device)


__all__ = get_module_objects(__name__)

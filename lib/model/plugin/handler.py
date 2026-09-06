#!/usr/bin/env python3
""" Handles Faceswap model loading, initialization, and configuration including weight initializers

This module provides the _ModelConfigure utility class for applying custom weight initializations
(ICNR, ConvolutionAware) and padding modifications (reflect mode) to model convolutions before
training begins. It also contains the FaceswapModel container class which manages plugin loading,
state persistence and configuration of neural network components during model setup
"""
from __future__ import annotations

import logging
import os
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.initializers import icnr, ConvolutionAware
from lib.utils import get_folder, get_module_objects

from plugins.plugin_loader import PluginLoader
from plugins.train import train_config as mod_cfg

from .legacy import KerasToTorch, save_migrated_state_dict
from .model_info import Info
from .state import State

if T.TYPE_CHECKING:
    from plugins.train.model.base import ModelPlugin
    from .model_info import Layer


logger = logging.getLogger(__name__)


class _ModelConfigure:
    """ Utility class for configuring model initialization and padding settings

    _ModelConfigure is a temporary configuration helper used only during model setup to apply
    custom weight initializations (ICNR or ConvolutionAware) and modify convolution padding modes
    from 'zeros' to 'reflect'. All effects are applied in-place to the model instance before
    training/inference begins and are dictated by the user config settings when initially creating
    the model. This options are applied globally so that plugins do not need to concern themselves
    with implementation details

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
            logger.debug("[_ModelConfigure] No custom initializers to apply")
            return
        # TODO prevent running on ImageNet weights load
        conv_aware = ConvolutionAware()
        icnr_conv = [x.name for v in self._model.info.structure.values()
                     if v.type == "PixelShuffle"
                     for x in self._get_prev_conv(v)] if self._init["icnr"] else []
        for k, v in model.named_modules():
            if k in icnr_conv and isinstance(v, nn.Conv2d):
                logger.debug("[_ModelConfigure] Applying ICNR Initialization: '%s' (%s)",
                             k, v.weight.shape)
                icnr(v.weight)
                if v.bias is not None:
                    nn.init.zeros_(v.bias)
            elif self._init["conv_aware"] and isinstance(v, nn.Conv2d):
                logger.info("[_ModelConfigure] Applying ConvAware Init '%s' %s...",
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
            logger.debug("[_ModelConfigure] No reflect padding to apply")
            return
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            pad = module.padding
            stride = module.stride
            kern = module.kernel_size
            if all(p == 0 for p in (pad if isinstance(pad, tuple) else (pad, pad))):
                logger.debug("[_ModelConfigure] Skip conv '%s' with zero padding: %s",
                             name, pad)
                continue
            if module.padding_mode != "zeros":
                logger.debug("[_ModelConfigure] Skip conv '%s' with non-zero padding: %s",
                             name, repr(module.padding_mode))
                continue
            if all(k == 1 for k in (kern if isinstance(kern, tuple) else (kern, kern))):
                logger.debug("[_ModelConfigure] Skip conv '%s' with kernel size == 1", name)
                continue
            if any(s > 1 for s in (stride if isinstance(stride, tuple) else (stride, stride))):
                logger.debug("[_ModelConfigure] Skip conv '%s' with stride > 1: %s",
                             name, stride)
                continue
            logger.debug("[_ModelConfigure] Reflect pad conv '%s'. padding: %s, kernel: %s, "
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
        logger.debug("[_ModelConfigure] Configured model")


class _ModelLoader:
    """ Utility class for loading and managing model checkpoint files

    Handles file I/O operations for Faceswap models including locating existing checkpoints,
    determining file existence states, and loading serialized weights from disk. Supports both
    modern PyTorch checkpoints (.ckpt/.pth) and legacy Keras format migration to maintain
    compatibility with older saved models. Automatically detects which save format
    is most recent and loads it accordingly during model initialization

    The loader tracks three file paths:
        1. ``model_name.ckpt``: Standard checkpoint (preferred, saves full state_dict)
        2. ``model_name.pth``: Just the model weights and state file. No training state
        3. Legacy paths - Keras files (.keras/.h5) for backward compatibility

    When no checkpoint exists, returns empty dictionary to signal fresh model creation needed.
    Handles weight migration from legacy Keras models when torch checkpoints are unavailable

    Parameters
    ----------
    model_name
        Model identifier string used in filenames and logging prefixes. Dashes preserved
        but converted for consistency with FaceswapModel naming conventions
    model_dir
        Directory path containing saved checkpoint files. Used to construct full paths for
        all supported file formats (.ckpt, .pth, .keras, .h5).

    Notes
    -----
    File existence checks prioritize .ckpt files first, then .pth as fallback. Legacy
    Keras files are only used when no torch checkpoint exists (migration scenario)

    The loader operates on absolute paths internally - model_dir is normalized to full
    path during __init__ via get_folder() utility function.
    """
    def __init__(self, model_name: str, model_dir: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = f"[{self.__class__.__name__}.{model_name}]"
        self._model_name = model_name
        self._model_dir = get_folder(model_dir)

        self._checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
        self._weights_path = os.path.join(model_dir, f"{model_name}.pth")
        self._legacy_paths = (os.path.join(model_dir, f"{model_name}.keras"),
                              os.path.join(model_dir, f"{model_name}.h5"))

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"model_name={self._model_name}, "
                f"model_dir={repr(self._model_dir)})")

    @property
    def _legacy_exists(self) -> bool:
        """ ``True`` if a legacy Keras format file (.keras or .h5) exists in model_dir """
        return any(os.path.isfile(x) for x in self._legacy_paths)

    @property
    def file_exists(self) -> bool:
        """ ``True`` if any non-legacy checkpoint or weight file exists in `model_dir` """
        return any(os.path.isfile(x) for x in (self._checkpoint_path, self._weights_path))

    @property
    def needs_upgrade(self) -> bool:
        """ ``True`` if a legacy Keras model exists but not a current torch model """
        return self._legacy_exists and not self.file_exists

    @property
    def checkpoint_path(self) -> str:
        """ The full path to the standard ``{model_name}.ckpt`` checkpoint file """
        return self._checkpoint_path

    def upgrade_legacy(self, model: FaceswapModel) -> tuple[ModelPlugin, dict[str, T.Any]]:
        """ Upgrade a legacy Keras model to a current Torch model and save to disk

        Parameters
        ----------
        model
            FaceswapModel instance required when migrating from Keras format. The converter needs
            a reference to the target plugin structure to properly map weights during migration
        """
        logger.info("[%s] Migrating weights from Keras model", self._model_name)

        migrator = KerasToTorch(next(f for f in self._legacy_paths if os.path.exists(f)))
        state = migrator.state
        model.state.load_state_dict(state)

        plugin = PluginLoader.get_model(self._model_name)(num_identities=2,
                                                          version=state["plugin_version"])
        migrator.migrate(plugin)
        # TODO optimizer/extra_state
        extra_state = {}

        model.plugin = plugin
        converted = model.state_dict()
        del model.plugin
        save_migrated_state_dict(converted, model.checkpoint_path)

        return plugin, extra_state

    def get_latest_save(self) -> str | None:
        """ Find and return path to most recently modified checkpoint file

        Compares modification times of available checkpoint files (.ckpt and .pth only, ignores
        legacy Keras files). Returns the filepath with highest mtime value, or ``None`` when
        neither standard checkpoint exists

        Returns
        -------
        Absolute path to most recent save file if checkpoints exist, otherwise ``None``. The
        returned file is either .ckpt (preferred) or .pth (fallback).

        Notes
        -----
        Legacy Keras files are explicitly excluded from comparison since they require migration and
        represent outdated model versions. Only torch-native checkpoints are considered for
        "latest" determination
        """
        if not self.file_exists:
            return None

        file_list = (self._checkpoint_path, self._weights_path)
        m_times = [os.path.getmtime(x) if os.path.isfile(x) else 0 for x in file_list]
        retval = file_list[m_times.index(max(m_times))]
        logger.debug("%s Latest save from %s: %s", self._name, file_list, retval)
        return retval

    def load(self) -> dict[str, T.Any]:
        """ Load state dictionary from checkpoint file or migrate legacy Keras weights

        Retrieves the most recent save and handles 2 loading paths:
            1. Checkpoint exists: Direct torch.load() with CPU map_location for weight transfer
            3. Checkpoint doesn't exist: Return empty dict to signal fresh model creation needed

        Returns
        -------
        Dictionary containing:
            - "version" : Plugin version string for compatibility checks
            - "state" : Model state with configuration parameters
            - "model" : PyTorch model weights (if checkpoint exists)
            - Extra training data items if load_extra_state=True

        Raises
        ------
        RuntimeError
            When legacy Keras model file is found but no torch structure reference provided,
            since migration requires knowing the target plugin architecture for weight mapping
        """
        filename = self.get_latest_save()
        if filename is None:
            logger.debug("%s No save files exist. Not loading", self._name)
            return {}

        state_dict = torch.load(filename, map_location="cpu", weights_only=True)
        logger.debug("Loaded model from disk: '%s'", filename)
        logger.debug("%s Loaded state_dict version %s. Keys: %s",
                     self._name, state_dict.get("version", 0.0), list(state_dict))
        return state_dict


class FaceswapModel:
    """ Container class for loading, managing, and configuring neural network models

    FaceswapModel wraps a ModelPlugin instance (neural network architecture) along with associated
    state management objects for persistence (State). It handles model creation from scratch,
    loading from saved checkpoints, plugin configuration (weight initialization, padding), and
    device placement during runtime. The class supports multiple identity models loaded
    simultaneously via the num_identities parameter which determines how many face encodings are
    processed per forward pass

    Loading Flow:
    -----------
    1. Loads config file if provided, instantiates plugin from PluginLoader
    2. Applies weight initializers (ICNR/ConvAware) and padding modifications
    3. Manages checkpoint serialization/deserialization

    State Management:
    -------------
    The State object tracks plugin version, configuration parameters (icnr_init, conv_aware_init),
    and model architecture details. _ModelLoader handles file I/O for loading checkpoints from
    disk. Extra state (non-standard attributes) can be persisted and managed separately via
    pop_extra_state().

    Parameters
    ----------
    name
        Model identifier string used to look up the plugin in PluginLoader.get_model(). Dashes are
        converted to underscores for internal attribute access consistency throughout the class
    model_dir
        Directory path containing saved model checkpoint files (model.pth, model.ckpt,
        config.json). Used by _ModelLoader to locate and load existing checkpoints
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
        self._num_identities = num_identities
        self._load_extra_state = load_extra_state
        self._conf_file = config_file
        self._log_name = f"[{self.__class__.__name__}.{name}]"
        self._info: Info | None = None

        self.name = name.replace("-", "_")
        """ Model identifier with dashes replaced by underscores for internal consistency """

        self.state = State(self.name)
        """ State management object tracking plugin version and configuration parameters """
        self.io = _ModelLoader(name, model_dir)  # TODO Private this
        """ File I/O handler for loading/saving checkpoints from model_dir directory """

        self.plugin: ModelPlugin
        """ The instantiated neural network module ready for training or inference after
        configuration completes """

        self.plugin, self._extra_state = self._load_plugin()

        self._configure()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = {"name": repr(self.name.replace("-", "_")),
                  "model_dir": repr(os.path.dirname(self.io.checkpoint_path)),
                  "num_identities": repr(self._num_identities),
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

    @property
    def latest_save(self) -> str | None:
        """ The absolute path to the most recently modified checkpoint file (.ckpt or .pth) or
        ``None`` if none exist """
        return self.io.get_latest_save()

    @property
    def checkpoint_path(self) -> str:
        """ The absolute path to the location where the checkpoint (.ckpt) may be saved """
        return self.io.checkpoint_path

    def _create_new(self) -> ModelPlugin:
        """ Instantiate a fresh plugin without loading any saved state

        Returns
        -------
        ModelPlugin
            Fresh plugin instance with random weights ready for training
        """
        logger.debug("[FaceswapModel] No state_dict to load. Creating new model")
        plugin = PluginLoader.get_model(self.name)(num_identities=self._num_identities)
        self.state.set_plugin_version(plugin.version)
        return plugin

    def _load_existing(self) -> tuple[ModelPlugin, dict[str, T.Any]]:
        """ Load existing plugin from checkpoint file including model weights and state
        dictionaries

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
            return self._create_new(), {}

        logger.info("%s Loading plugin from saved config", self._log_name)
        self.state.load_state_dict(T.cast(dict[str, T.Any], state_dict["state"]))
        plugin = PluginLoader.get_model(self.name)(num_identities=self._num_identities,
                                                   version=self.state.plugin_version)
        plugin.load_state_dict(state_dict["model"])
        extra_state = {k: v for k, v in state_dict.items()
                       if k not in ("state", "model", "version")} if self._load_extra_state else {}

        return plugin, extra_state

    def _load_plugin(self) -> tuple[ModelPlugin, dict[str, T.Any]]:
        """ Determine whether to create new or load existing plugin based on file existence check

        Returns
        -------
        plugin
            ModelPlugin instance loaded with weights from checkpoint file
        extra_state
            Dictionary of items (training data loaded from checkpoint) if load_extra_state=True
        """
        if self.io.needs_upgrade:
            return self.io.upgrade_legacy(self)

        if not self.io.file_exists:
            return self._create_new(), {}
        return self._load_existing()

    def _configure(self) -> None:
        """ Apply weight initializations and padding modifications after plugin instantiation """
        icnr_ = False if self.io.file_exists else self.state.config.get("icnr_init", False)
        conv = False if self.io.file_exists else self.state.config.get("conv_aware_init", False)
        reflect = self.state.config.get("reflect_padding", False)
        _ModelConfigure(self,
                        icnr_init=icnr_,
                        conv_aware_init=conv,
                        reflect_padding=reflect).configure()

    def state_dict(self) -> dict[str, T.Any]:
        """ Serialize model weights, plugin version, and state metadata for checkpoint saving

        Returns dictionary containing three keys:
            - "version" : Always 1.0 to indicate compatibility with current loader code
            - "model" : PyTorch model weights from self.plugin.state_dict()
            - "state" : State object data including plugin_version and configuration parameters

        This format is saved to checkpoint files for later restoration during training or inference
        sessions

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

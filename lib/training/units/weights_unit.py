#!/usr/bin/env python3
""" Handles loading external weights and freezing model layers for training

This optional module provides functionality to load pre-trained or fine-tuned weights from
external Torch model files (.pth/.ckpt) into the current Faceswap model, as well as freeze
certain layers during training. It includes two TrainingUnits:
  - LoadWeightsUnit: Loads external weights into specified layers of the current model
  - FreezeWeightsUnit: Freezes selected layers during training

Usage pattern: The LoadWeightsUnit is typically used to load a checkpoint from a previous
training or fine-tuned model, while FreezeWeightsUnit allows keeping parts of the network
frozen during subsequent training iterations to preserve learned features
"""
from __future__ import annotations

import logging
import os
import typing as T

import torch

from lib.logger import parse_class_init
from lib.utils import FaceswapError, get_module_objects

from .core import TrainingUnit

if T.TYPE_CHECKING:
    from lib.model.plugin import FaceswapModel
    from lib.training.training_loop import TrainStep

logger = logging.getLogger(__name__)


class LoadWeightsUnit(TrainingUnit):
    """ Training unit for loading external weights into the current model

    This unit loads pre-trained or fine-tuned weights from a Torch state dictionary file (.pth/
    .ckpt) and applies them to selected layers in the current Faceswap model. It validates
    compatibility between source and target models, extracts only requested layers, and handles
    partial matches gracefully by logging warnings rather than raising exceptions

    The unit is called once during initialization via on_load() after the model has been built,
    performing validation checks before loading weights to prevent corruption of existing model
    state. It works with any compatible PyTorch model architecture as long as layer naming
    conventions match between source and target models

    Parameters
    ----------
    weights_file
        Path to the Torch model file containing pre-trained or fine-tuned weights. Must be a .pth
        or .ckpt file (PyTorch state_dict format). The file should contain a dictionary with either
        "model" key for standard checkpoints or top-level keys matching layer names directly
    model
        The FaceswapModel instance whose layers will receive the loaded weights. Provides access to
        model plugin, state configuration, and naming conventions needed for compatibility checks

    Notes
    -----
    - Only layers specified in "load_layers" config option will be loaded
    - If a layer exists in source but not target (or vice versa), it is logged as warning
    - The unit performs validation before loading to ensure file integrity and compatibility
    - After successful load, training continues with the new weights replacing old ones
    """
    def __init__(self, weights_file: str, model: FaceswapModel) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._model_repr = repr(model)

        self._model_name = model.name
        self._plugin = model.plugin
        self._layers: list[str] = model.plugin.load_layers
        self._weights_file = weights_file

        self._validate()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"weights_file={self._weights_file}, "
                f"model={self._model_repr})")

    def _validate(self) -> None:
        """ Validate the weights file path and configuration settings

        Raises
        ------
        FaceswapError
            If the path to load weights from is invalid or no layers are selected to load
        """

        logger.debug("%s Validating: '%s'", self.log_name, self._weights_file)

        msg = ""
        if not self._layers:
            msg = "Load weights selected, but no layers have been selected in model config"
        elif not os.path.exists(self._weights_file):
            msg = f"Load weights selected, but the path '{self._weights_file}' does not exist."
        elif os.path.isdir(self._weights_file):
            msg = (f"The Load Weights path '{self._weights_file}' is a folder. It should be a "
                   ".pth/.ckpt model file.")
        elif os.path.splitext(self._weights_file)[-1].lower() not in (".pth", ".ckpt"):
            msg = (f"The Load Weights path '{self._weights_file}' is not a valid Torch model ("
                   ".pth/.ckpt) file.")

        if msg:
            msg += " Please check and try again."
            raise FaceswapError(msg)

    def _validate_model_file(self, model_weights: dict[str, T.Any]) -> None:
        """ Validate loaded weights dictionary for compatibility with current model + possibly warn

        Parameters
        ----------
        model_weights
            Dictionary containing loaded weights from torch.load(). Expected to be a Faceswap model
        """
        logger.debug("%s Validating imported weights: %s", self.log_name, list(model_weights))

        plugin_name = model_weights.get("state", {}).get("plugin_name")
        if not plugin_name:
            logger.warning("'%s' is not a Faceswap model file. Load weights will most likely fail",
                           self._weights_file)
            if "model" not in model_weights:  # Give us a slim chance to get something out of it
                model_weights["model"] = model_weights
        elif plugin_name != self._model_name:
            logger.warning("'%s' is a Faceswap '%s' model file. This is a '%s' model. Load "
                           "weights will most likely fail",
                           self._weights_file, plugin_name, self._model_name)

    def _get_layer_weights(self, model_weights: dict[str, T.Any]) -> dict[str, dict[str, T.Any]]:
        """ Extract and group weights by layer for selective loading

        Parameters
        ----------
        model_weights
            Dictionary containing loaded weights with structure like

        Returns
        -------
        Dictionary mapping layer names to their corresponding state dictionaries with filtered keys

        Raises
        ------
        FaceswapError
            If no weights dict found under "model" key or if none of the configured layers exist
        """
        weights: dict[str, T.Any] = model_weights.get("model", {})
        if not weights:
            raise FaceswapError(f"Could not find model weights in '{self._weights_file}'")

        retval = {x: {k[len(x) + 1:]: v for k, v in weights.items()
                      if k.startswith(f"{x}.")}
                  for x in self._layers}
        if not any(retval.values()):
            raise FaceswapError(
                f"These layers do not exist in '{self._weights_file}': {self._layers}"
                )

        missing = [k for k, v in retval.items() if not v]
        if missing:
            exists = [x for x in self._layers if x not in missing]
            logger.warning("These layers could not be loaded from weights file '%s': %s. The "
                           "following layer(s) will be loaded: %s",
                           self._weights_file, missing, exists)

        logger.debug("%s Got weights for layers: %s",
                     self.log_name, {k: list(v) for k, v in retval.items()})
        return retval

    def _load_weights(self, device: str) -> dict[str, dict[str, T.Any]]:
        """ Load weights from file and group by layer

        Parameters
        ----------
        device
            Device string (e.g., "cpu" or "cuda:0") used to relocate tensors during load

        Returns
        -------
        Dictionary mapping layer names to their state dictionaries ready for import into submodules
        """
        logger.debug("%s Loading + grouping %s layers from: '%s'",
                     self.log_name, self._layers, self._weights_file)
        fs_weights = torch.load(self._weights_file, map_location=device)
        self._validate_model_file(fs_weights)
        return self._get_layer_weights(fs_weights)

    def _import_weights(self, weights_by_layer: dict[str, T.Any]) -> None:
        """ Import filtered layer weights into corresponding model submodules

        Parameters
        ----------
        weights_by_layer
            Dictionary of layer names to their corresponding state dictionaries with filtered keys
        """
        # TODO test layers with shape mismatches + phaze-A
        # TODO test other models
        for layer, weights in weights_by_layer.items():
            mod = self._plugin.get_submodule(layer)
            logger.info("%s '%s' Loading weights: %s", self.log_name, layer, len(weights))
            mod.load_state_dict(weights)

    def on_load(self, loop: TrainStep) -> None:  # TODO move model to GPU later so we can do loading on CPU
        """ Load external weights and apply them to the current model

        Orchestrates the complete weight loading process by first loading weights from file with
        device specification from training loop, then grouping them by layer for selective import,
        and finally applying each group to corresponding submodules

        Parameters
        ----------
        loop
            The training step object that manages this unit's lifecycle
        """
        weights_by_layer = self._load_weights(str(loop.device))
        self._import_weights(weights_by_layer)
        logger.debug("%s Imported weights: %s", self.log_name, list(weights_by_layer))


class FreezeWeightsUnit(TrainingUnit):
    """ Training unit for freezing selected model layers during training

    This unit marks specific model layers as non-trainable by setting their requires_grad attribute
    to False. Freezing is useful when you want to keep certain pretrained components fixed while
    only training new or fine-tuning other parts of the network, preserving learned features from
    earlier training stages or external pre-training

    The unit validates that frozen layers exist in the current model architecture before attempting
    to freeze them. Layers are identified by their full module paths (e.g., "conv1" or
    "encoder.layer1.conv") and must match exactly with model.named_modules() output

    Parameters
    ----------
    model
        The FaceswapModel instance containing the plugin whose layers will be frozen. Provides
        access to state configuration including "freeze_layers" setting and model name for logging

    Notes
    -----
    - Only layers that exist in current model will be frozen (invalid names are skipped)
    - Frozen parameters continue forward pass but gradients won't update during backprop
    - Use with LoadWeightsUnit to freeze pretrained components while training rest of network
    """
    def __init__(self, model: FaceswapModel) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._repr_obj = f"{self.__class__.__name__}(model={model!r})"

        self._plugin = model.plugin
        self._name = model.name
        self._layers: list[str] = self._validate_layers(model.plugin.freeze_layers)

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return self._repr_obj

    def _validate_layers(self, layers: list[str]) -> list[str]:
        """ Validate and filter freeze layer names against actual model structure

        Parameters
        ----------
        layers
            The list of Torch nn.Module names that are selected for freezing

        Returns
        -------
        List of valid layer names that exist in current model and will be frozen
        """

        if not layers:
            logger.warning("%s `freeze_weights` selected but no layers specified to freeze. All "
                           "layers will be trainable.", self.log_name)
            return []

        selected = {x: [m[0] for m in self._plugin.named_modules()
                        if m[0] == x or m[0].startswith(f"{x}.")]
                    for x in layers}
        retval = []
        for k, v in selected.items():
            if not v:
                logger.warning("%s Layer '%s' set for freezing, but does not exist in '%s'. "
                               "Skipping",
                               self.log_name, k, self._name)
                continue
            logger.debug("%s Adding layer for freezing: '%s'", self.log_name, k)
            retval.append(k)

        if not retval:
            logger.warning("%s `freeze_weights` selected but no selected layers exist in '%s'. "
                           "All layers will be trainable.", self.log_name, self._name)

        logger.debug("%s Selected layers for freezing: %s", self.log_name, retval)
        return retval

    def on_load(self, loop: TrainStep) -> None:
        """ Freeze validated layers by setting requires_grad=False on their parameters

        Iterates through each configured frozen layer and finds all model parameters whose names
        start with that layer's path. For each parameter found, sets requires_grad to False to make
        it non-trainable. Logs info messages for each layer successfully frozen showing parameter
        count.

        Parameters
        ----------
        loop
            Training step object (not directly used but required by lifecycle contract)
        """
        for layer in self._layers:
            count = 0
            for name, param in self._plugin.named_parameters():
                if name.startswith(f"{layer}."):
                    param.requires_grad = False
                    count += 1
            if count > 0:
                logger.info("%s '%s' Parameters frozen: %s", self.log_name, layer, count)
        logger.debug("%s Frozen layers: %s", self.log_name, self._layers)


__all__ = get_module_objects(__name__)

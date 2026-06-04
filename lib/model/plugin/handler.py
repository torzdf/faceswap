#! /usr/env/bin/python3
"""Handlers for creating training and inference objects from Faceswap model plugins"""
from __future__ import annotations

import logging
import typing as T

import torch

from lib.logger import parse_class_init
from lib.training.optimizer import Optimizer
from lib.utils import get_module_objects

from plugins.plugin_loader import PluginLoader

from .train_state import State
from .saving import ModelIO

if T.TYPE_CHECKING:
    from plugins.train.model.base import ModelPlugin
    from plugins.train.train_config import Optimizer as opt_cfg


logger = logging.getLogger(__name__)


class FaceswapModel:
    """Holds the model and state information on a Faceswap model for serialization

    Parameters
    ----------
    name
        The name of the Faceswap model plugin to load
    num_identities
        The number of identities that the model is to be created for
    """
    def __init__(self, name: str, num_identities) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = f"[{self.__class__.__name__}.{name}]"

        self.name = name
        """The plugin name of the model to load"""
        self._num_identities = num_identities

        self._plugin = PluginLoader.get_model(name)(num_identities)
        self.state = State(self._plugin.__class__.__module__)

    def state_dict(self) -> dict[T.Literal["model", "state", "version"], float | dict[str, T.Any]]:
        """Get the Faceswap model's state_dict"""
        retval: dict[T.Literal["model", "state", "version"], float | dict[str, T.Any]] = {
            "version": 1.0,
            "model": self.plugin.state_dict(),
            "state": self.state.state_dict()
            }
        return retval

    @property
    def plugin(self) -> ModelPlugin:
        """The loaded Faceswap plugin"""
        return self._plugin

    def load_state_dict(self, state_dict: dict[T.Literal["model", "state", "version"],
                                               float | dict[str, T.Any]]) -> None:
        """Load the contents of the given state dict into this object. If a state key is provided
        within the state_dict then the model plugin will be re-initialized with the new settings.

        If no keys are provided the object remains unchanged

        Parameters
        ----------
        state_dict
            The Faceswap model's state_dict to load
        """
        if not state_dict:
            return

        logger.debug("%s version: %s, state_dict keys: %s",
                     self._name, state_dict.get("version", 0.0), list(state_dict))

        if "state" in state_dict:
            self.state.load_state_dict(T.cast(dict[str, T.Any], state_dict["state"]))
            logger.info("%s Reloading plugin", self._name)
            old = self._plugin
            self._plugin = old.__class__(self._num_identities)
            del old
        if "model" in state_dict:
            self._plugin.load_state_dict(T.cast(dict[str, T.Any], state_dict["model"]))

    def to(self, device: torch.Device) -> None:
        """Load the model and optimizer to the given device

        Parameters
        ----------
        device
            The device to load the model and optimizer to
        """
        logger.debug("%s Model to: %s", self._name, device)
        self.plugin.to(device)


class TrainHandler:
    """Handles the management of a Faceswap model plugin when training the model

    Parameters
    ----------
    name
        The name of the Faceswap model plugin to load
    num_identities
        The number of identities that the model is to be created for
    model_folder
        Full path to load/save model weights
    """
    def __init__(self,
                 name: str,
                 num_identities: int,
                 model_folder: str) -> None:
        logger.debug(parse_class_init(locals()))

        self.name = name
        """The name of the model plugin"""
        self._model = FaceswapModel(name, num_identities)
        self._io = ModelIO(self._model.name, model_folder)
        state_dict = self._io.load(model=self._model)
        self._model.load_state_dict({k: v for k, v in state_dict.items() if k != "optimizer"})
        self._optimizer_state = state_dict.get("optimizer")
        self._optimizer: Optimizer

    @property
    def model(self) -> ModelPlugin:
        """The currently loaded Faceswap Model"""
        return self._model.plugin

    @property
    def total_iterations(self) -> int:
        """The total number of iterations that the model has trained"""
        return self._model.state.iterations

    @property
    def session_id(self) -> int:
        """The current session ID. If training has not yet commenced, this will be the last session
        ID trained. If the first training step has been reached, this will be the currently
        training session ID"""
        return self._model.state.session_id

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer in use"""
        return self._optimizer

    def configure_model(self,
                        device: torch.Device,
                        optimizer_config: type[opt_cfg],
                        mixed_precision: bool,
                        warmup_steps: int) -> None:
        """Load optimize state and move to the correct device"""
        self._optimizer = Optimizer(self._model.plugin,
                                    optimizer_config,
                                    mixed_precision,
                                    warmup_steps)
        self._optimizer.load_state_dict(T.cast(dict[T.Literal["version", "optimizer", "scaler"],
                                                    float | T.Any],
                                               self._optimizer_state))
        self._model.plugin.train()
        self._model.to(device)
        self._optimizer.to(device)
        logger.debug("[Trainer] Configured model and trainer")

    def step(self, batch_size: int) -> None:
        """Update the iteration count in the state file

        Parameters
        ----------
        batch size that the plugin is training at. Used for creating a new session
        """
        self._model.state.step(batch_size)
        # TODO snapshot + backup here

    def save(self, with_optimizer: bool) -> None:
        """Save the model, state and optionally the optimizer

        Parameters
        ----------
        with_optimizer
            ``True`` to include the optimizer weights in the save file
        """
        state_dict = T.cast(dict[T.Literal["model", "state", "version", "optimizer"],
                                 float | dict[str, T.Any]], self._model.state_dict())
        if with_optimizer:
            state_dict |= {"optimizer": self._optimizer.state_dict()}
        self._io.save(state_dict)


__all__ = get_module_objects(__name__)

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

from .model_info import Info
from .saving import ModelIO
from .train_state import State

if T.TYPE_CHECKING:
    from lib.training.optimizer import Optimizer
    from lib.training.train import LossHandler
    from plugins.train.model.base import ModelPlugin
    from plugins.train.trainer.base import TrainerBase
    from .model_info import Layer


logger = logging.getLogger(__name__)


class FaceswapModel:
    """Holds the model and state information on a Faceswap model for serialization

    Parameters
    ----------
    name
        The name of the Faceswap model plugin to load
    num_identities
        The number of identities that the model is to be created for
    batch_size
        The batch size that the model is to be trained at, if opening for a training session,
        otherwise ``None``. Default: ``None``
    """
    def __init__(self, name: str, num_identities, batch_size: int | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = f"[{self.__class__.__name__}.{name}]"

        self.name = name
        """The plugin name of the model to load"""
        self._num_identities = num_identities

        self._plugin = PluginLoader.get_model(name)(num_identities)
        self.state = State(self._plugin.__class__.__module__, batch_size=batch_size)

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


class TrainConfigure:
    """Configures a Faceswap model for training based on user provided values

    Parameters
    ----------
    model_info
        The information about the loaded model's structure
    icnr_init
        ``True`` to initialize convolutions prior to up-scales with ICNR
    conv_aware_init
        ``True`` to apply conv_aware_init to all convolutions
    reflect_padding
        ``True`` to apply reflect padding to convolutions
    """
    def __init__(self,
                 model_info: Info,
                 icnr_init: bool,
                 conv_aware_init: bool,
                 reflect_padding: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._info = model_info
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
            return self._get_prev_conv(self._info.structure[lyr], collected)
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
        icnr_conv = [x.name for v in self._info.structure.values()
                     if v.type == "PixelShuffle"  # TODO all upscales?
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

    def configure(self, model: ModelPlugin) -> None:
        """Configure the given faceswap model with the user provided settings

        Parameters
        ----------
        model
            The Faceswap model to configure for training
        """
        self._apply_initializers(model)
        # TODO reflect padding
        # TODO MSG
        logger.debug("[Trainer] Configured model")


class TrainHandler:
    """Handles the management of a Faceswap model plugin when training the model

    Parameters
    ----------
    name
        The name of the Faceswap model plugin to load
    num_identities
        The number of identities that the model is to be created for
    batch_size
        The batch size that the model is to be trained at
    model_folder
        Full path to load/save model weights
    save_interval
        The number of steps between each model save
    snapshot_interval
        The number of steps between full model checkpoint snapshots
    """
    def __init__(self,
                 name: str,
                 num_identities: int,
                 batch_size: int,
                 model_folder: str,
                 save_interval: int,
                 snapshot_interval: int) -> None:
        logger.debug(parse_class_init(locals()))

        self.name = name
        """The name of the model plugin"""
        self.batch_size = batch_size
        """The batch size that is configured for training"""

        self._save_interval = save_interval
        self._snapshot_interval = snapshot_interval

        self._model = FaceswapModel(name, num_identities, batch_size=batch_size)
        self._io = ModelIO(self._model.name, model_folder)
        self._lrf_steps = 0

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
    def model_folder(self) -> str:
        """The folder that is being used to save the Faceswap model's weights"""
        return os.path.dirname(self._io.checkpoint_path)

    @property
    def model_exists(self) -> bool:
        """``True`` if a model weights file/checkpoint exists within the save folder"""
        return self._io.file_exists

    @property
    def optimizer(self) -> Optimizer:
        """The configured optimizer in use"""
        return self._optimizer

    def load_state_dict(self, optimizer: Optimizer | None) -> None:
        """Load the state from disk and set to the Model and State objects. Also loads Optimizer
        weights if one is attached to this object or provided here.

        Parameters
        ----------
        optimizer
            The configured Faceswap optimizer to additionally be loaded or ``None``. If an
            optimizer is already attached to this object it will be replaced with the given
            optimizer.
        """
        logger.info("[TrainHandler] Loading state_dict: %s", self._model)
        state_dict = self._io.load(model=self._model)
        self._model.load_state_dict({k: v for k, v in state_dict.items() if k != "optimizer"})
        if optimizer is not None:
            logger.debug("[TrainHandler] adding and Optimizer: %s", optimizer)
            self._optimizer = optimizer
        if hasattr(self, "_optimizer"):
            logger.debug("[TrainHandler] Loading optimizer state_dict: %s", self._optimizer)
            self._optimizer.load_state_dict(
                T.cast(dict[T.Literal["version", "optimizer", "scaler"],
                       float | dict[str, T.Any]],
                       state_dict.get("optimizer", {}))
            )

    def configure_model(self,
                        trainer_name: str,
                        model_info: Info,
                        mixed_precision: bool,
                        icnr_init: bool,
                        conv_aware_init: bool,
                        reflect_padding: bool,
                        device: torch.Device) -> TrainerBase:
        """Configure the model for training, applying any initialization and other post-build
        routines. Place the model onto the training device and return the object responsible for
        forward and backward passes through the model

        Parameters
        ----------
        trainer_name
            The name of the trainer plugin to use
        model_info
            The information about the loaded model's structure
        mixed_precision
            ``True`` for mixed precision training. ``False`` for full precision
        icnr_init
            ``True`` to initialize convolutions prior to up-scales with ICNR
        conv_aware_init
            ``True`` to apply conv_aware_init to all convolutions
        reflect_padding
            ``True`` to apply reflect padding to convolutions

        Returns
        -------
        The trainer plugin containing the configured model on the training device
        """
        is_new = not self._io.file_exists
        configurator = TrainConfigure(model_info,
                                      icnr_init and is_new,
                                      conv_aware_init and is_new,
                                      reflect_padding)
        configurator.configure(self.model)
        self._optimizer.to(device)
        self._model.to(device)
        self._model.plugin.train()
        retval = PluginLoader.get_trainer(trainer_name)(self._model.plugin,
                                                        self.batch_size,
                                                        mixed_precision,
                                                        str(device))
        if mixed_precision:  # TODO resume issues
            logger.info("Enabled Auto Mixed Precision")

        logger.debug("[TrainHandler] Configured model and trainer: %s", retval)
        return retval

    def _get_state_dict(self, with_optimizer: bool
                        ) -> dict[T.Literal["model", "state", "version", "optimizer"],
                                  float | dict[str, T.Any]]:
        """Obtain the latest model state dict

        Parameters
        ----------
        with_optimizer
            ``True`` to include the optimizer's state dict

        Returns
        -------
        The current faceswap model's state dict
        """
        retval = T.cast(dict[T.Literal["model", "state", "version", "optimizer"],
                             float | dict[str, T.Any]],
                        self._model.state_dict())
        if with_optimizer:
            retval |= {"optimizer": self._optimizer.state_dict()}
        return retval

    def set_lr_from_finder(self, value: float | None = None) -> bool:
        """Set the learning rate from a previous learning rate finder run

        Parameters
        ----------
        value
            If this is a value being loaded from the state_dict this should be ``None``. If it has
            been received from the learning rate finder, then it should be the value that will be
            stored, and the model set to

        Returns
        -------
        ``True`` if a previous LR finder rate was found and has been set. ``False`` if the LR
        finder has not been run for this model
        """
        if value is not None:
            # TODO Currently an issue with resuming MP. May not be LRF specific
            self.optimizer.disable_learning_rate_finder()
            self._model.state.lr_finder = value
            logger.debug("[TrainHandler] Restoring model weights")
            original_weights = torch.load(self._io.checkpoint_path)
            self._model.load_state_dict({"model": original_weights["model"]})
            logger.debug("[TrainHandler] Restoring optimizer weights")
            opt_state = {k: v for k, v in original_weights["optimizer"].items()
                         if k != "lrf_scheduler"}
            self._optimizer.load_state_dict(opt_state)

        lrf_rate = self._model.state.lr_finder
        if lrf_rate < 0:
            logger.debug("[TrainHandler] Learning rate finder has not been run. Not setting LR")
            return False
        logger.info("Setting learning rate from Learning Rate Finder: %s", f"{lrf_rate:.1e}")
        self.optimizer.set_lr(lrf_rate)
        self._model.state.learning_rate_from_finder = True
        return True

    def step(self, loss_handler: LossHandler, lrf_enabled: bool) -> bool:
        """Update the iteration count in the state file

        Parameters
        ----------
        loss_handler
            Holds the information about loss for the current save iteration. Reset on save
            iteration
        lrf_enabled
            ``True`` if the learning rate finder is enabled and running

        Returns
        -------
        ``True`` if the model was saved
        """
        if lrf_enabled:  # Just signal if preview would have been updated on a save interval
            self._lrf_steps += 1
            return self._lrf_steps % self._save_interval == 0

        self._model.state.step()

        retval = self._model.state.session_iterations % self._save_interval == 0
        if retval:
            self.save(loss_handler=loss_handler, is_exit=False)

        step = self._model.state.iterations
        if self._snapshot_interval != 0 and step % self._snapshot_interval == 0:
            state_dict = T.cast(
                dict[T.Literal["model", "state", "version", "optimizer"],
                     float | dict[str, T.Any]],
                self._model.state_dict() | {"optimizer": self._optimizer.state_dict()}
                )
            self._io.snapshot(step, state_dict)

        return retval

    def save(self, loss_handler: LossHandler | None, is_exit: bool = False) -> None:
        """Save the model, state and optionally the optimizer. Backup the last save if total
        average loss has dropped

        Parameters
        ----------
        loss_handler
            If this is part of the main training loop then this should be the loss handler, which
            is used to calculate if a backup should be made and resets the object for the next
            save iteration.
            If ``None`` then a full model checkpoint is made with no other action
            Holds the information about loss for the current save iteration. Is reset ons save
        is_exit
            ``True`` if save is being called on program exit
        """
        logger.debug("[TrainHandler] Saving. loss_handler: %s, is_exit: %s", loss_handler, is_exit)
        average_loss = None
        do_backup = False

        average_loss = 0.0
        if loss_handler is not None:
            average_loss = loss_handler.on_save()

            if self._model.state.lowest_avg_loss <= 0.0:
                self._model.state.lowest_avg_loss = average_loss

            if do_backup:
                self._io.backup()
                self._model.state.lowest_avg_loss = average_loss

        incl_optimizer = (loss_handler is None or
                          average_loss == 0.0 or
                          self.optimizer.save == "always" or
                          (is_exit and self.optimizer.save == "exit"))
        state_dict = self._get_state_dict(incl_optimizer)
        is_checkpoint = self._io.save(state_dict)

        msg = f"[Saved {'checkpoint' if is_checkpoint else 'model'}]"
        if average_loss != 0.0:
            msg += f" - Average loss since save: {average_loss:.5f}"
        if do_backup:
            msg += " [Model backed up]"
        logger.info(msg)


__all__ = get_module_objects(__name__)

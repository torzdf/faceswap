#! /usr/env/bin/python3
"""Handlers for creating training and inference objects from Faceswap model plugins"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.initializers import icnr, ConvolutionAware
from lib.torch_utils import get_device
from lib.training.loss import LossCollator
from lib.training.optimizer import Optimizer
from lib.utils import get_module_objects

from plugins.plugin_loader import PluginLoader

from .model_info import Info
from .saving import ModelIO
from .train_state import State

if T.TYPE_CHECKING:
    from plugins.train.model.base import ModelPlugin
    from plugins.train.trainer.base import TrainerBase
    from plugins.train.train_config import Loss as loss_cfg, Optimizer as opt_cfg
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


class TrainConfigure:
    """Configures a Faceswap model for training based on user provided values

    Parameters
    ----------
    model_info
        The information about the loaded model's structure
    loss_config
        The loss configuration options object
    optimizer_config
        The optimizer configuration options object
    icnr_init
        ``True`` to initialize convolutions prior to up-scales with ICNR
    conv_aware_init
        ``True`` to apply conv_aware_init to all convolutions
    mixed_precision
        ``True`` to train with mixed precision
    reflect_padding
        ``True`` to apply reflect padding to convolutions
    """
    def __init__(self,
                 model_info: Info,
                 loss_config: type[loss_cfg],
                 optimizer_config: type[opt_cfg],
                 icnr_init: bool,
                 conv_aware_init: bool,
                 mixed_precision: bool,
                 reflect_padding: bool) -> None:
        logger.debug(parse_class_init(locals()))

        self._info = model_info
        self._loss_cfg: type[loss_cfg] = loss_config
        self.optimizer_config: type[opt_cfg] = optimizer_config
        """The configuration options for the optimizer"""
        self.device = get_device()
        """The device that is training the model"""

        self._init = {"icnr": icnr_init, "conv_aware": conv_aware_init}
        self.mixed_precision = mixed_precision
        """``True`` if mixed precision is enabled."""
        self._reflect_padding = reflect_padding

    def _get_prev_conv(self, layer: Layer, collected: list[Layer] | None = None) -> list[Layer]:
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

    def _configure_loss(self, is_rgb: bool) -> LossCollator:
        """Configure the loss collator with the user selected loss functions and weights and
        copy it to the training device

        Parameters
        ----------
        is_rgb
            ``True`` if the model is training RGB. ``False`` for BGR

        Returns
        -------
        The collated loss functions for training the model
        """
        retval = LossCollator(
            functions=[self._loss_cfg.loss_function(),
                       self._loss_cfg.loss_function_2(),
                       self._loss_cfg.loss_function_3(),
                       self._loss_cfg.loss_function_4()],
            weights=[1.0,
                     self._loss_cfg.loss_weight_2() / 100.,
                     self._loss_cfg.loss_weight_3() / 100.,
                     self._loss_cfg.loss_weight_4() / 100.],
            color_order="rgb" if is_rgb else "bgr",
            use_mask=self._loss_cfg.penalized_mask_loss(),
            eye_multiplier=self._loss_cfg.eye_multiplier(),
            mouth_multiplier=self._loss_cfg.mouth_multiplier(),
            smallest_output=min(x[1] for x in self._info.output_shapes[0] if x[0] != 1),
            mask_loss=(None if not self._loss_cfg.learn_mask()
                       else self._loss_cfg.mask_loss_function()))
        retval.to(self.device)
        logger.debug("[TrainConfigure] loss: %s", retval)
        return retval

    def configure(self, model: ModelPlugin) -> LossCollator:
        """Configure the given faceswap model with the user provided settings

        Parameters
        ----------
        model
            The Faceswap model to configure for training

        Returns
        -------
        The configured, collated loss functions for training the model, on the training device
        """
        self._apply_initializers(model)
        # TODO loss y_true/pred switch
        # TODO reflect padding
        # TODO MSG
        loss = self._configure_loss(model.is_rgb)
        logger.debug("[Trainer] Configured model and loss: %s", loss)
        return loss


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
        self._loss: LossCollator

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
    def model_exists(self) -> bool:
        """``True`` if a model weights file/checkpoint exists within the save folder"""
        return self._io.file_exists

    @property
    def optimizer(self) -> Optimizer:
        """The configured optimizer in use"""
        return self._optimizer

    @property
    def loss(self) -> LossCollator:
        """The configured loss functions in use"""
        return self._loss

    def configure_model(self,
                        trainer_name: str,
                        train_config: TrainConfigure,
                        warmup_steps: int,
                        batch_size: int) -> TrainerBase:
        """Configure the model for training, applying any initialization and other post-build
        routines. Obtain the loss function and optimizer and return the training plugin

        Parameters
        ----------
        trainer_name
            The name of the trainer plugin to use
        train_config
            The user training configuration options
        warmup_steps
            The number of steps to warmup the learning rate
        batch_size
            The batch size to train the model

        Returns
        -------
        The trainer plugin containing the configured model on the training device
        """
        self._loss = train_config.configure(self.model)
        self._optimizer = Optimizer(self._model.plugin,
                                    train_config.optimizer_config,
                                    train_config.mixed_precision,
                                    warmup_steps)
        self._optimizer.load_state_dict(T.cast(dict[T.Literal["version", "optimizer", "scaler"],
                                                    float | T.Any],
                                               self._optimizer_state))
        self._optimizer.to(train_config.device)
        self._model.to(train_config.device)
        self._model.plugin.train()
        retval = PluginLoader.get_trainer(trainer_name)(self._model.plugin,
                                                        batch_size,
                                                        train_config.mixed_precision,
                                                        str(train_config.device))
        logger.debug("[TrainHandler] Configured model and trainer: %s", retval)
        return retval

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

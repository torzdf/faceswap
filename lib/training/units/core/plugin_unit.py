#! /usr/env/bin/python3
""" Training unit for managing model plugin operations during training

This module contains the core PluginUnit class which is responsible for handling the core model
training operations including data loading, trainer execution, loss calculation, and optimization
steps. It serves as the primary interface  between the training loop and the actual model
processing components
"""
from __future__ import annotations

import logging
import typing as T

import torch

from lib.logger import parse_class_init
from lib.training.loss import LossCollator
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    from lib.training.data import TrainLoader
    from lib.training.loss import BatchLoss
    from lib.training.training_loop import TrainStep
    from lib.model.plugin import FaceswapModel
    from plugins.train.trainer.base import TrainerPlugin
    from .optimizer_unit import OptimizerUnit


logger = logging.getLogger(__name__)


class PluginUnit(TrainingUnit):
    """ Manages the core model training operations through a Faceswap Plugin

    This unit handles the execution of the trainer plugin each training iteration by fetching
    batches from the loader, moving tensors to the appropriate device, and calling the trainer's
    step method which handles the full forward/backward/optimization cycle. Loss values are tracked
    via a property that exposes the current batch loss for use by other units like LossUnit.

    Parameters
    ----------
    loader
        The TrainLoader instance that provides input data (images, targets, metadata)
    trainer
        The TrainerPlugin that executes the training step (forward pass, loss calculation,
        backward pass, and optimizer update)
    optimizer
        The optimizer unit managing parameter updates
    model
        The FaceswapModel object containing the neural network, state and info for the model
    device
        The device (CPU/GPU) to run training on
    loss_functions
        Mapping of loss function names to their respective weights
    penalize_mask_loss
        Whether to apply mask-based penalties during loss calculation
    eye_multiplier
        Multiplier for eye-related losses
    mouth_multiplier
        Multiplier for mouth-related losses
    mask_loss
        The type of mask loss function to use, if applicable
    """
    def __init__(self,  # pylint:disable=too-many-arguments,too-many-positional-arguments
                 loader: TrainLoader,
                 trainer: TrainerPlugin,
                 optimizer: OptimizerUnit,
                 model: FaceswapModel,
                 device: torch.Device,
                 loss_functions: dict[str, float],
                 penalize_mask_loss: bool,
                 eye_multiplier: float,
                 mouth_multiplier: float,
                 mask_loss: str | None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._extra_repr = {"model": repr(model),
                            "loss_functions": repr(loss_functions),
                            "penalize_mask_loss": repr(penalize_mask_loss),
                            "eye_multiplier": repr(eye_multiplier),
                            "mouth_multiplier": repr(mouth_multiplier),
                            "mask_loss": repr(mask_loss)}
        self._loader = loader
        self._trainer = trainer
        self._optimizer = optimizer
        self._device = device

        self._state = model.state
        self._loss_fn = self._configure_loss(loss_functions,
                                             penalize_mask_loss,
                                             eye_multiplier,
                                             mouth_multiplier,
                                             mask_loss,
                                             model.info.output_shapes,
                                             model.plugin.is_rgb)

        self._current_loss: list[BatchLoss] = []

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = {k: self._extra_repr[k] if k in self._extra_repr else repr(self.__dict__[f"_{k}"])
                  for k in ("loader",
                            "trainer",
                            "optimizer",
                            "model",
                            "device",
                            "loss_functions",
                            "penalize_mask_loss",
                            "eye_multiplier",
                            "mouth_multiplier",
                            "mask_loss")}
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @property
    def current_loss(self) -> list[BatchLoss]:
        """ The detached loss values from the most recently processed training batch """
        return self._current_loss

    def _configure_loss(self,
                        loss_functions: dict[str, float],
                        penalize_mask_loss: bool,
                        eye_multiplier: float,
                        mouth_multiplier: float,
                        mask_loss: str | None,
                        output_shapes: list[list[tuple[int, int, int]]],
                        is_rgb: bool) -> LossCollator:
        """ Configure and initialize the composite loss function for training

        Parameters
        ----------
        loss_functions
            Mapping of loss function names to their respective weights
        penalize_mask_loss
            Whether to apply mask-based penalties during loss calculation
        eye_multiplier
            Multiplier for eye-related losses
        mouth_multiplier
            Multiplier for mouth-related losses
        mask_loss
            The type of mask loss function to use, if applicable
        output_shapes
            The shapes of model outputs for configuration purposes
        is_rgb
            Whether the color channel order is RGB (True) or BGR (False)

        Returns
        -------
        Configured loss collator instance
        """
        loss = LossCollator(
            functions=list(loss_functions),
            weights=list(loss_functions.values()),
            color_order="rgb" if is_rgb else "bgr",
            use_mask=penalize_mask_loss,
            eye_multiplier=eye_multiplier,
            mouth_multiplier=mouth_multiplier,
            smallest_output=min(x[1] for x in output_shapes[0] if x[0] != 1),
            mask_loss=mask_loss)
        logger.debug("%s Configured loss: %s", self.log_name, loss)
        return loss

    def on_load(self, loop: TrainStep) -> None:
        """ Initialize the model and loss function on the specified device

        Moves the trainer model and loss functions to the configured training device

        Parameters
        ----------
        loop
            The training step object that manages this unit's lifecycle
        """
        logger.debug("%s Model and loss to : %s", self.log_name, self._device)
        self._trainer.model.to(self._device)
        self._loss_fn.to(self._device)

    def step(self, iteration: int) -> None:
        """ Execute one training step on a batch of data

        This method:
            - Processes the next batch from the loader
            - Moves all tensors to the appropriate device (GPU/CPU)
            - Runs the forward pass through the trainer
            - Calculates loss using configured functions
            - detaches the loss tensors from the computation graph
            - Performs backwards pass to update the optimizer
            - Detaches and stores the loss for the current iteration in the `current_loss`
            parameter

        Parameters
        ----------
        iteration
            The current iteration number in the training process
        """
        inputs, targets, meta = next(self._loader)
        self._current_loss.clear()
        loss = self._trainer.step([i.to(self._device) for i in inputs],
                                  [t.to(self._device) for t in targets],
                                  meta.to(self._device),
                                  self._loss_fn,
                                  self._optimizer)
        self._current_loss.extend([x.detach() for x in loss])


__all__ = get_module_objects(__name__)

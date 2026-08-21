#! /usr/env/bin/python3
""" Handles the processing of the forward and backwards passes through the Faceswap plugin """
from __future__ import annotations

import logging
import typing as T

import torch

from lib.logger import parse_class_init
from lib.training.loss import LossCollator
from lib.utils import get_module_objects

from . import TrainingUnit

if T.TYPE_CHECKING:
    from lib.training.data import TrainLoader
    from lib.training.loss import BatchLoss
    from lib.training.training_loop import TrainStep
    from lib.model.plugin.handler import FaceswapModel
    from plugins.train.trainer.base import TrainerPlugin
    from .optimizer_unit import OptimizerUnit


logger = logging.getLogger(__name__)


class PluginUnit(TrainingUnit):
    """ Handles processing of forward and backward passes through a Faceswap plugin

    This unit orchestrates each training iteration by fetching batches from the loader, moving
    tensors to the appropriate device, and calling the trainer's step method which handles the full
    forward/backward/optimization cycle. Loss values are tracked via a property that exposes the
    current batch loss for use by other units like LossUnit.

    Parameters
    ----------
    loader
        The TrainLoader instance that provides input data (images, targets, metadata)
    trainer
        The TrainerPlugin that executes the training step (forward pass, loss calculation,
        backward pass, and optimizer update)
    optimizer
        The Optimizer used to update model weights after each backward pass
    model
        The FaceswapModel object containing the neural network and info for the model (RGB mode,
        output shapes)
    device
        The torch device that the model will be trained on
    loss_functions
        Dictionary mapping loss names to their weights (e.g., {"L1": 1.0, "MSE": 0.5})
    penalize_mask_loss
        Whether to apply mask penalty during loss computation
    eye_multiplier
        Multiplier for eye region loss component
    mouth_multiplier
        Multiplier for mouth region loss component
    mask_loss
        Type of mask loss function to use (e.g., "L1", "MSE") or None if disabled

    Notes
    -----
    This unit is called once per training iteration. It does not handle saving or other lifecycle
    events - those are managed by separate units in the TrainStep.

    The current_loss property provides access to loss values computed during this step,
    which are used by LossUnit for tracking averages and reporting metrics.
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
        """ String representation for debugging and logging """
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
        """ A list of BatchLoss objects containing the loss outputs for each identity
        processed during this iteration. Used by LossUnit for averaging and reporting metrics.

        Notes
        -----
        Values are populated after trainer.step() and detached at the start of loss_unit.step().
        """
        return self._current_loss

    def _configure_loss(self,
                        loss_functions: dict[str, float],
                        penalize_mask_loss: bool,
                        eye_multiplier: float,
                        mouth_multiplier: float,
                        mask_loss: str | None,
                        output_shapes: list[list[tuple[int, int, int]]],
                        is_rgb: bool) -> LossCollator:
        """Configure and initialize the composite loss function for training

        Parameters
        ----------
        loss_functions
            Dictionary mapping loss names to their weights (e.g., {"L1": 1.0, "MSE": 0.5})
        penalize_mask_loss
            Whether to apply mask penalty during loss computation
        eye_multiplier
            Multiplier for eye region loss component
        mouth_multiplier
            Multiplier for mouth region loss component
        mask_loss
            Type of mask loss function to use (e.g., "L1", "MSE") or None if disabled
        output_shapes
            List of output shapes from the model's decoder layers
        is_rgb
            Whether the model uses RGB color format

        Returns
        -------
        The collated loss functions ready for computing gradients during each training step

        Notes
        -----
        This method creates a LossCollator that combines multiple loss components into a single
        trainable objective. Each component can have its own weight, allowing fine-grained control
        over which aspects of the face swap quality are prioritized during training.
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

    def on_start(self, loop: TrainStep) -> None:
        """ Initialize device reference and move model and loss functions to training device

        Parameters
        ----------
        loop
            The active TrainStep instance. Used to access the shared device context.

        Notes
        -----
        This method is called before training begins to establish references to the training
        loop's device and maintain proper lifecycle tracking within the unit system. It also moves
        the loss function to the appropriate device.
        """
        logger.debug("%s Model and loss to : %s", self.log_name, self._device)
        self._trainer.model.to(self._device)
        self._loss_fn.to(self._device)

    def step(self, iteration: int) -> None:
        """ Execute a single training iteration through the plugin

        Fetches the next batch from the data loader, moves all tensors to the appropriate device
        (GPU/CPU), and invokes the trainer's step method which handles the complete forward-
        backward optimization cycle. The resulting loss values are detached from the gpu, cleared
        from any previous state and then populated with the new loss metrics for use by other
        units.

        Parameters
        ----------
        iteration
            The current total iteration count for logging purposes

        Notes
        -----
        This is called once per training iteration by the TrainStep. Loss values computed during
        this step are stored in the ``current_loss`` property for use by LossUnit and other
        dependent units.
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

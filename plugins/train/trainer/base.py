#!/usr/bin/env python3
"""Base Class for Faceswap Trainer plugins. All Trainer plugins should be inherited from
this class.

At present there is only the :class:`~plugins.train.trainer.original` plugin, so that entirely
inherits from this class. If further plugins are developed, then common code should be kept here,
with "original" unique code split out to the original plugin.
"""
from __future__ import annotations
import abc
import logging
import typing as T
from contextlib import nullcontext

import torch
from torch.cuda import OutOfMemoryError

from lib.logger import parse_class_init
from lib.utils import FaceswapError, get_module_objects


if T.TYPE_CHECKING:
    from lib.training.data import BatchMeta
    from lib.training.loss import LossCollator, BatchLoss
    from lib.training.units.core import OptimizerUnit
    from plugins.train.model.base import ModelPlugin

logger = logging.getLogger(__name__)


class TrainerPlugin(abc.ABC):
    """A trainer plugin interface. It must implement the method "train_batch" which takes an input
    of inputs to the model and target images for model output. It returns loss per side

    Parameters
    ----------
    model
        The configured Faceswap model plugin to be trained
    batch_size
        The batch size to train the model at
    """
    def __init__(self, model: ModelPlugin, batch_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self.model: ModelPlugin = model
        """The model plugin to be trained"""
        self.batch_size = batch_size
        """The batch size for each iteration to be trained through the model."""
        self.sampler = self.get_sampler()
        """The data sampler that the data loader should use"""
        self._forward_context = nullcontext()
        self._name = f"[{self.__class__.__name__}]"

    @abc.abstractmethod
    def get_sampler(self) -> type[torch.utils.data.RandomSampler |
                                  torch.utils.data.DistributedSampler]:
        """Override to set the sampler that the Torch DataLoader should use

        Returns
        -------
        The sampler that the torch DataLoader should use
        """

    @abc.abstractmethod
    def backward(self, loss: torch.Tensor, optimizer: OptimizerUnit) -> None:
        """Override to run a single backward pass through the model for a single batch

        Parameters
        ----------
        loss
            The loss scalar to use for backprop
        optimizer
            The configured optimizer to use for backprop
        """

    @abc.abstractmethod
    def forward(self,
                inputs: list[torch.Tensor],
                targets: list[torch.Tensor],
                meta: BatchMeta,
                loss_func: LossCollator) -> list[BatchLoss]:
        """Override to run a single forward pass through the model for a single batch

        Parameters
        ----------
        inputs
            The batch of input image tensors to the model of length(num inputs)
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch
        loss_func
            The configured loss functions to use
        """

    def set_training_precision(self, mixed_precision: bool, device_type: str) -> None:
        """ Set the precision that the model should be trained at

        Parameters
        ----------
        mixed_precision
            ``True`` to used mixed precision training, ``False`` to use full precision
        device_type
            The torch device type that is training the model
        """
        if mixed_precision:
            logger.info("%s Enabled Mixed Precision training", self._name)
            self._forward_context = torch.autocast(device_type=device_type, dtype=torch.float16)
        else:
            logger.debug("%s Using Full Precision training", self._name)

    def step(self,
             inputs: list[torch.Tensor],
             targets: list[torch.Tensor],
             meta: BatchMeta,
             loss_func: LossCollator,
             optimizer: OptimizerUnit) -> list[BatchLoss]:
        """Runs the plugin's forward and backwards pass through the model for a single batch

        Parameters
        ----------
        inputs
            The batch of input image tensors to the model of length(num inputs)
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch
        loss_func
            The selected loss functions for the model
        optimizer
            The configured Optimizer to use

        Returns
        -------
        The loss for each input to the model in order (A, B, ...) on the training device
        """
        try:
            logger.trace(  # type:ignore[attr-defined]
                "%s Forward step. inputs: %s, targets: %s, meta: %s, loss_func: %s",
                self._name, [x.shape for x in inputs],
                [t.shape for t in targets],
                meta,
                loss_func
                )
            with self._forward_context:
                loss = self.forward(inputs, targets, meta, loss_func)

            total_loss = T.cast(torch.Tensor, sum(x.total for x in loss))
            logger.trace("%s Backward step. loss: %s, optimizer: %s",  # type:ignore[attr-defined]
                         self._name, total_loss, optimizer)

            self.backward(total_loss, optimizer)
            logger.trace("%s Step complete. loss: %s",  # type:ignore[attr-defined]
                         self._name, loss)
        except OutOfMemoryError as err:
            msg = ("You do not have enough GPU memory available to train the selected model at "
                   "the selected settings. You can try a number of things:"
                   "\n1) Close any other application that is using your GPU (web browsers are "
                   "particularly bad for this)."
                   "\n2) Lower the batchsize (the amount of images fed into the model each "
                   "iteration)."
                   "\n3) Try enabling 'Mixed Precision' training."
                   "\n4) Use a more lightweight model, or select the model's 'LowMem' option "
                   "(in config) if it has one.")
            raise FaceswapError(msg) from err
        return loss


__all__ = get_module_objects(__name__)

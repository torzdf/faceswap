#!/usr/bin/env python3
"""Original Trainer """
from __future__ import annotations

import logging
import typing as T

import torch

from lib.utils import get_module_objects
from .base import TrainerBase

if T.TYPE_CHECKING:
    from lib.training.data import BatchMeta
    from lib.training.loss import BatchLoss, LossCollator
    from lib.training.optimizer import Optimizer


logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    """Original trainer"""

    def get_sampler(self) -> type[torch.utils.data.RandomSampler]:
        """Obtain a standard random sampler

        Returns
        -------
        The Random sampler
        """
        return torch.utils.data.RandomSampler

    def forward(self,
                inputs: list[torch.Tensor],
                targets: list[torch.Tensor],
                meta: BatchMeta,
                loss_func: LossCollator) -> list[BatchLoss]:
        """Perform the forward pass on the model

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
            The configured loss function's collator

        Returns
        -------
        The loss for each input to the model in order (A, B, ...)
        """
        predictions: list[list[torch.Tensor]] = self.model(inputs)
        num_sides = len(inputs)
        losses = [loss_func(predictions[i], [t[:, i] for t in targets], meta[i])
                  for i in range(num_sides)]
        return losses

    def backward(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
        """Perform the backwards pass on the model

        Parameters
        ----------
        loss
            The total loss scalar from the forward pass
        optimizer
            The configured Optimizer to use
        """
        optimizer.backward(loss)


__all__ = get_module_objects(__name__)

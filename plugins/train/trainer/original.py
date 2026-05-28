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
    from lib.training.loss import BatchLoss
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

    def _forward(self,
                 inputs: list[torch.Tensor],
                 targets: list[torch.Tensor],
                 meta: BatchMeta) -> list[BatchLoss]:
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

        Returns
        -------
        The loss for each input to the model in order (A, B, ...)
        """
        num_sides = len(inputs)

        identity_inputs = identity_targets = None
        if self.config.identity_loss:
            num_sides //= 2
            identity_inputs = inputs[-num_sides:]
            inputs = inputs[:-num_sides]
            identity_targets = targets.pop(-1)

        predictions = self.model.model(inputs, training=True)
        num_outputs = len(predictions) // num_sides

        swap_predictions: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_sides)}
        swap_dissim_indices: dict[int, list[int]] = {i: [] for i in range(num_sides)}
        if identity_inputs is not None and identity_targets is not None:
            # TODO it would be more efficient to re-use the encoder outputs from the original
            # predictions and just roll the down-streams from that point rather than feeding the
            # entire model again. May be complex to do with our plugin architecture though and not
            # worth investigating whilst models are still in keras

            # Only need final image output for identity loss so get final image index per side here
            image_idx = next(i for i in reversed(range(num_outputs))
                             if predictions[:num_outputs][i].shape[-1] != 1)

            dsm_targets = list(range(num_sides))
            for shift in range(1, num_sides):
                rolled = identity_inputs[shift:] + identity_inputs[:shift]
                swap_pred = self.model.model(rolled, training=True)
                logger.trace("[Trainer] swap outputs for side %s: %s",  # type:ignore[attr-defined]
                             shift, [x.shape for x in swap_pred])
                for inp_idx, dsm_tgt in enumerate(dsm_targets[shift:] + dsm_targets[:shift]):
                    swap_predictions[inp_idx].append(swap_pred[inp_idx * num_outputs + image_idx])
                    swap_dissim_indices[inp_idx].append(dsm_tgt)
            logger.trace(  # type:ignore[attr-defined]
                "[Trainer] Swap predictions: %s, swap_dissim_indices: %s",
                {k: [x.shape for x in v] for k, v in swap_predictions.items()},
                swap_dissim_indices)

        losses: list[BatchLoss] = [
            self.loss_func(y_true_all=[t[:, i] for t in targets],
                           y_pred_all=predictions[i * num_outputs:i * num_outputs + num_outputs],
                           side_index=i,
                           meta=meta,
                           swap_true=identity_targets,
                           swap_pred=swap_predictions[i],
                           dissim_indices=swap_dissim_indices[i])
            for i in range(num_sides)
            ]

        logger.trace("[Original] Losses: %s", losses)  # type:ignore[attr-defined]
        return losses

    def _backwards_and_apply(self, loss: list[BatchLoss], optimizer: Optimizer) -> None:
        """Perform the backwards pass on the model

        Parameters
        ----------
        loss
            The loss for each output from the model
        optimizer
            The configured Optimizer to use
        """
        total_loss = T.cast(torch.Tensor, sum(x.total for x in loss))
        optimizer.backward(total_loss)
        optimizer.step()

    def train_batch(self,
                    inputs: list[torch.Tensor],
                    targets: list[torch.Tensor],
                    optimizer: Optimizer,
                    meta: BatchMeta) -> list[BatchLoss]:
        """Run a single forward and backwards pass through the model for a single batch

        Parameters
        ----------
        inputs
            The batch of input image tensors to the model of length(num inputs)
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        optimizer
            The configured Optimizer to use
        meta
            The meta information for the batch

        Returns
        -------
        The loss for each input to the model in order (A, B, ...)
        """
        loss = self._forward(inputs, targets, meta)
        self._backwards_and_apply(loss, optimizer)
        return loss


__all__ = get_module_objects(__name__)

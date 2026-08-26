#! /usr/bin/env python3
"""Training unit for managing loss calculation and monitoring during training

This module contains the core LossUnit class which is responsible for tracking, calculating, and
reporting training losses throughout the model training process. It handles NaN protection, average
loss computation, and loss contribution  analysis to provide meaningful feedback during training
"""
from __future__ import annotations

import logging
import time
import typing as T

import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import FaceswapError, get_module_objects

from lib.training.data import get_label

from .base import TrainingUnit

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.training.loss import BatchLoss

logger = logging.getLogger(__name__)


class LossUnit(TrainingUnit):
    """ Manages loss calculations, monitoring, and reporting during training iterations

    This unit tracks loss values from training batches, computes running averages, provides NaN
    protection to prevent corrupted models, and outputs detailed loss information for monitoring
    progress. It also calculates loss contributions for different components to help understand
    model performance.

    Parameters
    ----------
    nan_protection
        Whether to enable NaN detection and automatic termination during training
    current_loss
        Reference to the list storing current batch loss values. The list persists, so it will
        always contain the loss for the current step
    device
        The device (CPU/GPU) on which loss calculations will be performed
    """
    def __init__(self,
                 nan_protection: bool,
                 current_loss: list[BatchLoss],
                 device: torch.Device) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._nan_protection = nan_protection
        self._loss = current_loss
        self._device = device

        self._averages: dict[T.Literal["unweighted", "weighted"], dict[str, torch.Tensor]] = {}
        self._loss_count = 0
        self._current_average: npt.NDArray[np.float32] = np.array(0.0).astype("float32")

    def __repr__(self) -> str:
        """ String representation for debugging and logging """
        return (f"{self.__class__.__name__} ("
                f"nan_protection={self._nan_protection!r}, "
                f"current_loss={self._loss!r}, "
                f"device={self._device!r})")

    @property
    def current_average(self) -> npt.NDArray[np.float32]:
        """ The average loss value since the last save operation. A 0-d array that is updated every
        step so a reference to this object can be safely taken """
        return self._current_average

    def _reset_averages(self, names: list[str] | None = None) -> None:
        """ Reset internal averaging structures to start fresh calculations

        Parameters
        ----------
        names
            List of loss component names to reset averages for. If None, uses current names
        """
        names = list(self._averages["unweighted"]) if names is None else names
        self._averages = {w: {k: torch.zeros((1, ), dtype=torch.float32, device=self._device)
                              for k in names}
                          for w in ("unweighted", "weighted")}
        self._loss_count = 0
        logger.debug("%s Reset loss averages: %s", self.log_name, self._averages)

    def _handle_nan(self, loss: list[BatchLoss]) -> None:
        """ Check for and handle NaN values in loss calculations

        Parameters
        ----------
        loss
            The current batch of loss values to check for NaN

        Raises
        ------
        FaceswapError
            When a NaN value is detected and nan_protection is enabled
        """
        if not self._nan_protection:
            return
        if all(torch.isfinite(val.total).all() for val in loss):
            return

        loss_str = ", ".join(f"Loss {get_label(i, len(loss))}: {round(x.total.item(), 6)}"
                             for i, x in enumerate(loss))
        msg = f"NaN Detected. {loss_str}"
        failed = ", ".join(f"{key}({get_label(i, len(loss))})"
                           for i, out in enumerate(loss)
                           for unweighted in out.unweighted
                           for key, sub_loss in unweighted.items()
                           if not torch.isfinite(sub_loss).all())
        if failed:
            msg += f". The loss function(s) that NaN'd: {failed}"
        logger.critical(msg)
        raise FaceswapError("A NaN was detected and you have NaN protection enabled. Training "
                            "has been terminated.")

    def _update_averages(self, loss: list[BatchLoss]) -> None:
        """ Computes weighted and unweighted running averages for each loss since the last save

        Parameters
        ----------
        loss
            The current batch of loss values to use for updating averages
        """
        contrib = [x.get_contributions() for x in loss]
        totals = {w: {k: sum(d[w][k] for d in contrib) for k in contrib[0][w]} for w in contrib[0]}
        if not self._averages:
            self._reset_averages(names=list(totals["unweighted"]))
        self._loss_count += 1
        self._averages = {a: {k: self._averages[a][k] + (totals[a][k] -
                                                         self._averages[a][k]) / self._loss_count
                              for k in self._averages[a]}
                          for a in self._averages}

    def _print_loss(self, loss: list[BatchLoss], iteration: int) -> None:
        """ Format and print the current loss values to console for real-time monitoring

        Parameters
        ----------
        loss
            The current batch of loss values to print
        iteration
            The current training iteration number
        """
        totals = {i: x.total.item() for i, x in enumerate(loss)}
        output = ", ".join(f"Loss {get_label(k, len(totals))}: {v:.5f}"
                           for k, v in totals.items())
        timestamp = time.strftime("%H:%M:%S")
        output = f"[{timestamp}] [#{iteration:05d}] {output}"
        print(f"{output}", end="\r")

    def step(self, iteration: int) -> None:
        """ Process loss calculations and updates for the current iteration

        Handles:
            - NaN checking and early exit (if enabled)
            - Updating the running average since the last save
            - Outputting the loss for this iteration to console

        Notes
        -----
        This step is skipped during pre-training phase

        Parameters
        ----------
        iteration
            The current training iteration number
        """
        if iteration < 0:  # Loss does not get updated during pre-train
            logger.trace("%s Pre-training. Not handling loss",  # type:ignore[attr-defined]
                         self.log_name)
            return

        self._handle_nan(self._loss)
        self._update_averages(self._loss)
        self._print_loss(self._loss, iteration)

    def _output_contributions(self) -> None:
        """ Output detailed loss contribution ratios to the logging system since last save """
        totals = {w: sum(m.values()) for w, m in self._averages.items()}
        ratios = {w: {k: round(((v / totals[w]) * 100.).item(), 1) for k, v in m.items()}
                  for w, m in self._averages.items()}
        msg = "Ratios since save [Weighted (Unweighted)]: "
        msg += ", ".join(f"{k}: {ratios['weighted'][k]}% ({ratios['unweighted'][k]}%)"
                         for k in ratios["unweighted"])
        logger.info(msg)

    def on_save(self, iteration: int) -> None:
        """ Perform loss calculations on a save iteration

        This method:
            - Calculates and logs the percentage contributions of each loss component to provide
            insight into which parts of the model are most influential
            - Updates the `current_average` property with the average loss since the last save
            iteration
            - Resets the running averages dictionary to prepare for the next save iteration

        Parameters
        ----------
        iteration : int
            The current training iteration number when the save is triggered
        """
        logger.debug("%s on save step %s", self.log_name, iteration)
        if not self._averages:
            logger.debug("%s No save averages to collate", self.log_name)
            return
        self._output_contributions()
        self._current_average[...] = T.cast(torch.Tensor,
                                            sum(self._averages["weighted"].values())).item()
        self._reset_averages()
        logger.debug("%s Average total since last save: %s", self.log_name, self._current_average)


__all__ = get_module_objects(__name__)

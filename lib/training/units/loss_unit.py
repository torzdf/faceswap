#! /usr/env/bin/python3
""" Handles the processing of loss function outputs each batch """
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
    """ Class to handle the stepping of loss functions

    This unit processes loss function outputs for each batch during training, tracking
    running averages and handling NaN protection. It interfaces with TensorBoard logging
    through the current_average property object.

    Parameters
    ----------
    nan_protection
        ``True`` to enable NaN detection and automatic termination if a loss value becomes
        non-finite during training
    current_loss
        The list that will hold the BatchLoss objects containing the loss outputs for each identity
        processed during this iteration. The list persists, so it will always contain the loss for
        the current step.
    device
        The torch device that the model will be trained on
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

        self._loss: list[BatchLoss]  # set in on_start

    def __repr__(self) -> str:
        """ String representation for debugging and logging """
        return (f"{self.__class__.__name__} ("
                f"nan_protection={self._nan_protection!r}, "
                f"current_loss={self._loss!r}, "
                f"device={self._device!r})")

    @property
    def current_average(self) -> npt.NDArray[np.float32]:
        """ Get the average loss for the last save iteration

        This property provides access to the computed average loss value since the last
        model save, used for tracking model performance and backup decisions.

        Returns
        -------
        The averaged loss value as a 0-dimensional arrat representing total loss across all
        components. The array is updated directly each step so that a reference to this object
        can safely be taken

        Notes
        -----
        This value is updated when on_save() is called and reset after each save operation.
        """
        return self._current_average

    def _reset_averages(self, names: list[str] | None = None) -> None:
        """ Reset the loss averages to zero and initialize tracking for specified components

        Parameters
        ----------
        names
            The name of the loss functions to track when initially setting up. If ``None``
            all currently tracked losses are used. Default: ``None``

        Notes
        -----
        This method is called both at initialization and after each save operation to reset
        tracking for the next interval.
        """
        names = list(self._averages["unweighted"]) if names is None else names
        self._averages = {w: {k: torch.zeros((1, ), dtype=torch.float32, device=self._device)
                              for k in names}
                          for w in ("unweighted", "weighted")}
        self._loss_count = 0
        logger.debug("[%s Reset loss averages: %s", self.log_name, self._averages)

    def _handle_nan(self, loss: list[BatchLoss]) -> None:
        """ Handle NaN values detected in loss outputs and terminate training if protection is
        enabled

        Raises
        ------
        FaceswapError
            If a NaN is detected with nan_protection enabled. The error message includes details
            about which loss function(s) failed, formatted for easy debugging.

        Notes
        -----
        When nan_protection is disabled (False), this method silently continues training despite
        NaN values. This allows experimentation but may lead to unstable models.
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
        """ Store the total running weighted and unweighted averages for each loss function for the
        current save iteration

        Parameters
        ----------
        loss
            The list of detached loss outputs on the training device in order (A, B, ...)

        Notes
        -----
        Uses exponential moving average smoothing to track loss values across iterations.
        This provides a smooth curve that reflects recent performance while reducing noise
        from individual batch fluctuations.
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
        """ Outputs the loss values for the current iteration to the console

        Parameters
        ----------
        loss
            The detached loss output from the model on the training device in order (A, B, ...)
        iteration
            The total training step being processed

        Notes
        -----
        Output is formatted with timestamp and iteration number for monitoring progress.
        Uses carriage return to overwrite the line as iterations complete, providing a
        cleaner terminal output experience.
        """
        totals = {i: x.total.item() for i, x in enumerate(loss)}
        output = ", ".join(f"Loss {get_label(k, len(totals))}: {v:.5f}"
                           for k, v in totals.items())
        timestamp = time.strftime("%H:%M:%S")
        output = f"[{timestamp}] [#{iteration:05d}] {output}"
        print(f"{output}", end="\r")

    def step(self, iteration: int) -> None:
        """ Handle the logging and processing of loss values for a training batch

        Parameters
        ----------
        iteration
            The current total training iteration. Is -1 if training has not fully started
            (eg during learning rate warmup)

        Notes
        -----
        Processing steps:

        1. Detaches loss from the model
        2. Handle NaNs if protection is enabled
4       3. Track running averages since last save
        4. Print loss to console if learning rate finder is disabled

        The iteration count of -1 indicates initialization phase where no processing occurs.
        """
        if iteration < 0:  # TODO LRF check
            return

        loss = [x.detach() for x in self._loss]
        self._handle_nan(loss)
        self._update_averages(loss)
        self._print_loss(loss, iteration)

    def _output_contributions(self) -> None:
        """ Output the loss function contributions since the last save

        Notes
        -----
        Calculates both weighted and unweighted percentages for each loss component, helping
        identify which aspects of the model are contributing most to overall training error.
        Useful for debugging loss function balancing issues.
        """
        totals = {w: sum(m.values()) for w, m in self._averages.items()}
        ratios = {w: {k: round(((v / totals[w]) * 100.).item(), 1) for k, v in m.items()}
                  for w, m in self._averages.items()}
        msg = "Ratios since save [Weighted (Unweighted)]: "
        msg += ", ".join(f"{k}: {ratios['weighted'][k]}% ({ratios['unweighted'][k]}%)"
                         for k in ratios["unweighted"])
        logger.info(msg)

    def on_save(self, iteration: int) -> None:
        """ Logging actions to perform when the model is saved

        Parameters
        ----------
        iteration
            The total iteration number for the model

        Notes
        -----
        Actions performed in order:

        1. Log debug message with save iteration count
        2. If no averages exist (initialization), skip processing
        3. Output contribution ratios for monitoring
        4. Calculate total average loss across all weighted components
        5. Reset running averages to zero for next interval

        The current_average property is updated so other units can access this value.
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

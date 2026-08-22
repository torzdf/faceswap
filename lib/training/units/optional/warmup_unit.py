#!/usr/bin/env python3
""" Handles learning rate warmup scheduling during training initialization.

This optional module provides components for gradually increasing the learning rate from zero to
its target value over a specified number of steps, preventing unstable early training behavior when
the model hasn't yet converged on proper weights. It integrates with PyTorch's LRScheduler API to
ensure compatibility with standard optimizer interfaces
"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch.optim.lr_scheduler import LRScheduler

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from lib.training.units.core import TrainingUnit

if T.TYPE_CHECKING:
    from torch.optim import Optimizer
    from lib.training.training_loop import TrainStep

logger = logging.getLogger(__name__)


class WarmupScheduler(LRScheduler):
    """ PyTorch learning rate scheduler for warmup period

    This scheduler linearly increases the learning rate from zero to its target value over a
    specified number of steps, preventing unstable early training behavior when the model hasn't
    yet converged on proper weights. After warmup completes, it maintains the constant target
    learning rate. It integrates with PyTorch's LRScheduler API to ensure compatibility with
    standard optimizer interfaces

    Parameters
    ----------
    optimizer
        The Torch optimizer whose learning rate will be adjusted during warmup
    steps
        Number of warmup steps over which the learning rate will be increased from 0 to target value
    last_epoch, optional
        The index of the last epoch. Default: ``-1``
    """
    def __init__(self, optimizer: Optimizer, steps: int, last_epoch: int = -1) -> None:
        logger.debug(parse_class_init(locals()))
        self.steps = steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        """ Calculate learning rate for current epoch
        
        Returns
        -------
        A list containing learning rate values for each parameter group in the optimizer
        """
        if self.last_epoch >= self.steps:
            return self.base_lrs

        factor = self.last_epoch / self.steps
        lrs = [base_lr * factor for base_lr in self.base_lrs]
        logger.trace("Learning rate set to %s for step %s/%s",  # type:ignore[attr-defined]
                     lrs, self.last_epoch, self.steps)
        return lrs


class WarmupUnit(TrainingUnit):
    """ Learning rate warmup unit for training initialization

    This unit manages the learning rate warmup process during the initial phase of training,
    gradually increasing the learning rate from zero to its target value over a specified number
    of steps. This prevents unstable early training behavior when the model hasn't yet converged
    on proper weights. Detailed logging is provided at set reporting intervals (every 10% of warmup
    period plus start/end points).

    It automatically stops applying warmup logic once ``iteration > warmup_steps``, allowing
    normal training to proceed without interference.

    Parameters
    ----------
    warmup_steps
        Number of steps over which to increase the learning rate from 0 to target value
    """
    def __init__(self, warmup_steps: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._warmup_steps = warmup_steps
        self._reporting_points = [int(warmup_steps * i / 10) for i in range(11)]

        self._scheduler: WarmupScheduler  # set in on_start
        self._optimizer: Optimizer  # set in on_start

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}(warmup_steps={self._warmup_steps!r})")

    @classmethod
    def _fmt(cls, value: float) -> str:
        """ Format float value in scientific notation
        
        Parameters
        ----------
        value
            The numeric learning rate or other metric to format.

        Returns
        -------
        String representation in scientific notation with one decimal place (eg: ``"1e-03"``)
        """
        return f"{value:.1e}"

    def on_start(self, loop: TrainStep) -> None:
        """ Initialize the warmup scheduler with optimizer

        Sets up the warmup scheduler using the Torch optimizer from the training loop

        Parameters
        ----------
        loop
            The training step object that manages this unit's lifecycle
        """
        self._optimizer = loop.optimizer_unit.optimizer

    def _report_progress(self, iteration: int) -> None:
        """ Log learning rate warmup progress at predefined reporting points
        
        Parameters
        ----------
        iteration
            The current training iteration number. Logging only occurs if this value is valid
        """
        if iteration not in ([1] + self._reporting_points + [self._warmup_steps]):
            return

        current_lr = T.cast(float, self._scheduler.get_last_lr()[0])
        target_lr = T.cast(float, self._scheduler.base_lrs[0])

        if iteration == 1:
            logger.info("[Learning Rate Warmup] Start: %s, Target: %s, Steps: %s",
                        self._fmt(current_lr), self._fmt(target_lr), self._warmup_steps)
            return
        if iteration == self._warmup_steps:
            logger.info("%s Final Learning Rate: %s", self.log_name, self._fmt(target_lr))
            return

        progress = int(round(100 / (len(self._reporting_points) - 1) *
                       self._reporting_points.index(iteration), 0))
        logger.info("[Learning Rate Warmup] Step: %s/%s (%s), Current: %s, Target: %s",
                    iteration,
                    self._warmup_steps,
                    f"{progress}%",
                    self._fmt(current_lr),
                    self._fmt(target_lr))

    def step(self, iteration: int) -> None:
        """ Execute one training step for warmup operations

        Handles the warmup scheduling by incrementing the learning rate at each iteration during
        the warmup phase and reporting progress. After warmup period completes, simply returns
        without performing any operations

        Parameters
        ----------
        iteration
            Current training iteration number. Negative values indicate pre-training phase
        """
        if iteration < 0:  # TODO need to make sure this doesn't still kick in
            logger.trace("%s Pre-training. Not handling warmup",  # type:ignore[attr-defined]
                         self.log_name)
            return
        if iteration > self._warmup_steps:
            return
        self._scheduler.step()
        self._report_progress(iteration)


__all__ = get_module_objects(__name__)

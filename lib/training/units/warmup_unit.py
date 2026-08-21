#!/usr/bin/env python3
""" Handles learning rate warmup scheduling during training initialization.

This module provides components for gradually increasing the learning rate from zero to its
target value over a specified number of steps, preventing unstable early training behavior when
the model hasn't yet converged on proper weights. It integrates with PyTorch's LRScheduler API
to ensure compatibility with standard optimizer interfaces.
"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch.optim.lr_scheduler import LRScheduler

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .core import TrainingUnit

if T.TYPE_CHECKING:
    from torch.optim import Optimizer
    from lib.training.training_loop import TrainStep

logger = logging.getLogger(__name__)


class WarmupScheduler(LRScheduler):
    """ Custom learning rate scheduler that linearly warms up the LR from 0 to target.

    This scheduler implements a linear warmup schedule where the learning rate increases
    proportionally with each step until reaching the full target value after ``steps`` iterations.
    After warmup completes, it maintains the constant target learning rate. It extends PyTorch's
    base LRScheduler API to integrate seamlessly with standard optimizer interfaces.

    The linear ramp ensures stable training during early epochs when model weights are still
    being initialized. This is particularly important for models trained from scratch where
    sudden high learning rates can cause divergence.

    Parameters
    ----------
    optimizer
        The PyTorch Optimizer instance to schedule the learning rate for. Must have a valid
        ``lr`` attribute that will be updated each step.
    steps
        The total number of warmup iterations before reaching target learning rate. LR will
        equal 0 at step 0 and full value at step=steps.
    last_epoch
        Index of the last epoch (or -1 for fresh scheduler). Default: ``-1``

    Attributes
    ----------
    steps
        The total number of warmup iterations configured in ``__init__``
    base_lrs
        Base learning rates before scheduling modifications (set by parent class)

    Notes
    -----
    During the warmup phase (steps < self.steps), the scheduler computes:

    .. code-block:: python

        lr = base_lr * (step / steps)

    where ``base_lr`` is the target learning rate. This linear interpolation provides a smooth
    transition from zero to full learning rate. After warmup completes, it returns ``base_lrs``
    unchanged.

    Examples
    --------
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> scheduler = WarmupScheduler(optimizer, steps=1000)
    >>> for step in range(1500):
    ...     train_step()
    ...     scheduler.step()  # Linear LR increase to 0.001 over first 1000 steps
    """
    def __init__(self, optimizer: Optimizer, steps: int, last_epoch: int = -1) -> None:
        logger.debug(parse_class_init(locals()))
        self.steps = steps
        """ The total number of warmup iteration """
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        """ Compute the current learning rate(s).

        Returns a list of learning rates that linearly ramp from 0 to ``base_lrs`` over the
        warmup period, then maintain at target value.

        Returns
        -------
        A list containing learning rate values for each parameter group in the optimizer.
        During warmup (step < self.steps), returns ``base_lr * (last_epoch / steps)``.
        After warmup completes, returns ``base_lrs`` unchanged.

        Notes
        -----
        This method is called by the parent LRScheduler class at each step to determine what
        learning rate to apply. The implementation checks if warmup has completed before
        returning the appropriate value.

        Examples
        --------
        >>> # During warmup: lr = base_lr * (current_step / total_steps)
        >>> scheduler.get_lr()  # Returns list with scaled LRs
        """
        if self.last_epoch >= self.steps:
            return self.base_lrs

        factor = self.last_epoch / self.steps
        lrs = [base_lr * factor for base_lr in self.base_lrs]
        logger.trace("Learning rate set to %s for step %s/%s",  # type:ignore[attr-defined]
                     lrs, self.last_epoch, self.steps)
        return lrs


class WarmupUnit(TrainingUnit):
    """ Manages learning rate warmup during the initial training phase.

    This unit handles the gradual increase of learning rate from zero to target value over a
    specified number of iterations. It integrates with PyTorch's LRScheduler API and provides
    detailed logging at set reporting intervals (every 10% of warmup period plus start/end points).

    The unit is responsible for:

    - Initializing the learning rate scheduler during training startup
    - Stepping the scheduler each iteration to apply warmed-up LR values
    - Logging progress information at key milestones (start, end, and 10% intervals)

    It automatically stops applying warmup logic once ``iteration > warmup_steps``, allowing
    normal training to proceed without interference.

    Notes
    -----
    The unit integrates with the training lifecycle as follows:

    - **on_start**: Creates a WarmupScheduler attached to the optimizer
    - **step**: Applies scheduler step each iteration, logs progress at reporting points
    - After warmup completes (iteration > _warmup_steps), skips scheduling logic
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
        """ Format a floating point value for logging output.

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
        """ Initialize the learning rate scheduler for warmup period.

        This method is called at training startup and creates a WarmupScheduler attached to
        the optimizer's parameter groups. The scheduler will linearly increase LR from zero
        to target over ``self._warmup_steps`` iterations.

        Parameters
        ----------
        loop
            The TrainStep instance providing access to the current optimizer object. This is
            used to retrieve and configure the underlying PyTorch Optimizer.

        Notes
        -----
        Creates a WarmupScheduler with:

        - **optimizer**: Retrieved from ``loop.optimizer._optimizer`` (PyTorch's internal storage)
        - **steps**: Set to ``self._warmup_steps`` from initialization

        The scheduler integrates seamlessly with PyTorch's optimizer.step() cycle.

        Examples
        --------
        >>> # Called at training start:
        >>> unit.on_start(training_loop)  # Creates WarmupScheduler
        """
        self._optimizer = loop.optimizer_unit.optimizer

    def _report_progress(self, iteration: int) -> None:
        """ Log learning rate warmup progress at predefined reporting points.

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
        """ Apply warmup scheduler step and log progress if due.

        Called each training iteration to advance the learning rate schedule by one step.
        After warmup period completes, simply returns without performing any operations.

        Parameters
        ----------
        iteration
            The current total iteration being processed. Used to determine if we're still in
            warmup phase and at which reporting points to log progress.

        Notes
        -----
        This method performs two actions:

        1. **Schedule advancement**: Advances the internal learning rate calculation for the next
        iteration

        2. **Progress reporting**: Logs information at predefined intervals

        The warmup phase ends when iteration exceeds self._warmup_steps, after which this
        method becomes a no-op (early returns).
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

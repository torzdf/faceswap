#!/usr/bin/env python3
"""Learning Rate Finder for faceswap.py."""
from __future__ import annotations
import logging
import os
import shutil
import typing as T
from datetime import datetime
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from torch import Tensor
    from lib.model.plugin.handler import TrainHandler
    from lib.training.train import Trainer

logger = logging.getLogger(__name__)


class LRStrength(Enum):
    """Enum for how aggressively to set the optimal learning rate"""
    DEFAULT = 10
    AGGRESSIVE = 5
    EXTREME = 2.5


class LearningRateFinder:  # pylint:disable=too-many-instance-attributes
    """Learning Rate Finder

    Parameters
    ----------
    enabled
        ``True`` if LRF has been enabled. ``False`` if disabled
    trainer
        The configured and loaded training pipeline
    steps
        The number of steps to run the finder for
    strength
        How aggressively to set the optimal learning rate
    mode
        The mode to run the Learning Rate Finder in
    stop_factor
        When to stop finding the optimal learning rate
    beta
        Amount to smooth loss by
    """
    def __init__(self,
                 enabled: bool,
                 trainer: Trainer,
                 steps: int,
                 strength: T.Literal["default", "aggressive", "extreme"],
                 mode: T.Literal["set", "graph_and_set", "graph_and_exit"],
                 stop_factor: int = 4,
                 beta: float = 0.98) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = "[LearningRateFinder]"

        self.is_enabled = self._on_launch(enabled, trainer._model_handler)
        """``True`` if LRF has been enabled. ``False`` if disabled"""

        if not self.is_enabled:
            logger.debug("%s Disabled. Exiting early", self._name)
            return

        self._trainer = trainer
        self._steps = steps
        self._strength = LRStrength[strength.upper()].value
        self._mode = mode
        self._stop_factor = stop_factor
        self._beta = beta

        self._model_handler = trainer._model_handler

        self._losses: list[float | Tensor] = []
        self._learning_rates: list[float] = []
        self._loss: dict[T.Literal["avg", "best"], float | Tensor] = {"avg": 0.0, "best": 1e9}
        self._best_lr: None | float = None

        self._scheduler = trainer.optimizer.enable_learning_rate_finder(steps, 1e-10, 1e-1)

        logger.info("Finding learning rate...")
        self._p_bar = tqdm(range(1, self._steps + 1),
                           desc="Current: N/A      Best: N/A    ",
                           leave=False)

    @property
    def best_lr(self) -> None | float:
        """The discovered best learning rate or ``None`` if not found"""
        return self._best_lr

    def _on_launch(self, enabled: bool, model_handler: TrainHandler) -> bool:
        """Process the LR Finder on startup.

        If not enabled just return ``False``.
        If a previous LRF run has found a learning rate, sets the model's LR to this value and
        returns ``False``.
        If this is a new model than return ``True``
        Otherwise this is a resumed model with no LR detected, so return ``False``

        Parameters
        ----------
        enabled
            ``True`` if the Learning Rate Finder is enabled. ``False`` if it is disabled

        Returns
        -------
        ``True`` if the learning rate finder should run in the training loop, otherwise ``False``
        """
        if not enabled:
            logger.debug("%s Disabled", self._name)
            return False

        if model_handler.set_lr_from_finder():
            logger.debug("%s Disabling as LR set from previous Finder", self._name)
            return False

        # TODO check for state resume. Will have implications for existence of state file
        if model_handler.total_iterations > 0 or model_handler.session_id > 0:
            logger.debug("%s Disabled as not new model", self._name)
            return False
        return True

    def _on_batch_end(self, loss: Tensor) -> bool:
        """Learning rate actions to perform at the end of a batch

        Parameters
        ----------
        loss
            The total loss value for the current batch

        Returns
        -------
        ``True`` if training should cease. ``False`` to continue
        """
        if torch.isnan(loss):
            logger.info("Loss has NaN'd. Exiting early")
            return True

        self._learning_rates.append(T.cast(float, self._scheduler.get_last_lr()[0]))
        self._loss["avg"] = (self._beta * self._loss["avg"]) + ((1 - self._beta) * loss)
        smoothed = self._loss["avg"] / (1 - (self._beta ** self._scheduler.last_epoch))
        self._losses.append(smoothed)

        stop_loss = self._stop_factor * self._loss["best"]
        if self._scheduler.last_epoch > 1 and smoothed > stop_loss:
            logger.info("Loss has diverged. Exiting early")
            return True

        if self._scheduler.last_epoch == 1 or smoothed < self._loss["best"]:
            self._loss["best"] = smoothed

        if self._scheduler.last_epoch == self._steps:
            logger.debug("[LearningRateFinder] Reached final step. Exiting")
            return True

        return False

    def _plot_loss(self, skip_begin: int = 10, skip_end: int = 1) -> None:
        """Plot a graph of loss vs learning rate and save to the training folder

        Parameters
        ----------
        skip_begin
            Number of iterations to skip at the start. Default: `10`
        skip_end
            Number of iterations to skip at the end. Default: `1`
        """
        if self._mode not in ("graph_and_set", "graph_and_exit"):
            return

        matplotlib.use("Agg")
        lrs = self._learning_rates[skip_begin:-skip_end]
        losses = T.cast(list[float], self._losses[skip_begin:-skip_end])
        plt.plot(lrs, losses, label="Learning Rate")
        best_idx = self._losses.index(self._loss["best"])
        best_lr = self._learning_rates[best_idx]
        for val, color in zip(LRStrength, ("g", "y", "r")):
            l_r = best_lr / val.value
            idx = lrs.index(next(r for r in lrs if r >= l_r))
            plt.plot(l_r, losses[idx],
                     f"{color}o",
                     label=f"{val.name.title()}: {l_r:.1e}")

        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.legend()

        now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        output = os.path.join(self._model_handler.model_folder, f"learning_rate_finder_{now}.png")
        logger.info("Saving Learning Rate Finder graph to: '%s'", output)
        plt.savefig(output)

    def _finalize(self) -> bool:
        """Clean up and perform post finder actions

        Returns
        -------
        ``True`` if training should exit. ``False`` to continue
        """
        print("\x1b[2K", end="\r")  # Clear line
        self._losses = [x.item() if isinstance(x, torch.Tensor) else x for x in self._losses]
        best_idx = self._losses.index(self._loss["best"])
        new_lr = self._learning_rates[best_idx] / self._strength
        if new_lr < 1e-9:
            logger.error("The optimal learning rate could not be found. This is most likely "
                         "because you did not run the finder for enough iterations.")
            shutil.rmtree(self._model_handler.model_folder)  # TODO
            return True

        self._best_lr = new_lr
        self._plot_loss()
        self._model_handler.set_lr_from_finder(new_lr)
        del self._losses
        self.is_enabled = False
        return self._mode == "graph_and_exit"

    def _update_progress_bar(self, amount: int | None = None) -> None:
        """Update the description and count of the progress bar for the current iteration

        Parameters
        ----------
        amount
            The amount to iterate the progress bar by. Default: ``None`` (1 step)
        """
        current = self._learning_rates[-1]
        best_idx = self._losses.index(self._loss["best"])
        best = self._learning_rates[best_idx] / self._strength
        self._p_bar.update(1 if amount is None else amount)
        self._p_bar.set_description(f"Current: {current:.1e}  Best: {best:.1e}")

    def step(self, loss: Tensor) -> bool:
        """Perform a Learning Rate Finder step

        Parameters
        ----------
        loss
            The total loss scalar from the latest forward pass

        Returns
        -------
        ``True`` if Process should exit. ``False`` to keep running
        """
        if self._scheduler.last_epoch == 1:  # Need to have populated optimizer weights on step 1
            logger.debug("%s Saving initial weights", self._name)
            self._model_handler.save(None)

        if self._on_batch_end(loss):
            self._p_bar.close()
            return self._finalize()

        self._update_progress_bar()
        return False


__all__ = get_module_objects(__name__)

#!/usr/bin/env python3
"""Learning Rate Finder for faceswap.py."""
from __future__ import annotations
import logging
import os
import typing as T
from datetime import datetime
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from torch import Tensor
    from torch.optim.optimizer import Optimizer
    from lib.model.plugin.handler import TrainHandler

logger = logging.getLogger(__name__)


class LRStrength(Enum):
    """Enum for how aggressively to set the optimal learning rate"""
    DEFAULT = 10
    AGGRESSIVE = 5
    EXTREME = 2.5


class LRFScheduler(ExponentialLR):
    """A scheduler that expands on ExponentialLR Scheduler to capture loss history for the Learning
    Rate Finder duration

    When last_epoch=-1, sets initial lr as lr.

    Parameters
    ----------
    optimizer
        Wrapped optimizer.
    gamma
        Multiplicative factor of learning rate decay.
    beta
        Amount to smooth loss by
    total_steps
        The number of steps to run the finder for
    last_epoch
        The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 gamma: float,
                 beta: float,
                 total_steps: int,
                 last_epoch: int = -1
                 ) -> None:
        self.beta = beta
        self.total_steps = total_steps
        self.losses: list[float | Tensor] = []
        self.learning_rates: list[float] = []
        self.loss: dict[T.Literal["avg", "best"], float | Tensor] = {"avg": 0.0, "best": 1e9}
        super().__init__(optimizer, gamma, last_epoch)

    def state_dict(self) -> dict[str, T.Any]:
        """Obtain the state dict for this scheduler"""
        retval = super().state_dict()
        retval["beta"] = self.beta
        retval["total_steps"] = self.total_steps
        retval["losses"] = self.losses
        retval["learning_rates"] = self.learning_rates
        return retval

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """Load the state dict for the scheduler"""
        self.beta = state_dict.pop("beta")
        self.total_steps = state_dict.pop("total_steps")
        self.losses = state_dict.pop("losses")
        self.learning_rates = state_dict.pop("learning_rates")
        super().load_state_dict(state_dict)

    def step(self, epoch: int | None = None) -> None:
        """Step the scheduler.

        Parameters
        ----------
        epoch
            .. deprecated:: 1.4
            If provided, sets :attr:`last_epoch` to ``epoch`` and uses :meth:`_get_closed_form_lr`
            if it is available. This is not universally supported. Use :meth:`step` without
            arguments instead.

        Note
        ----
        Call this method after calling the optimizer's :meth:`~torch.optim.Optimizer.step`.
        """
        super().step(epoch=epoch)
        if self.last_epoch > 0:
            self.learning_rates.append(T.cast(float, self.get_last_lr()[0]))

    def track_loss(self, loss: Tensor) -> float | torch.Tensor:
        """Track the latest lost values for the current step

        Parameters
        ----------
        loss
            The total loss scalar for the current LRF step

        Returns
        -------
        The smoothed loss value
        """
        self.loss["avg"] = (self.beta * self.loss["avg"]) + ((1 - self.beta) * loss)
        smoothed = self.loss["avg"] / (1 - (self.beta ** self.last_epoch))
        self.losses.append(smoothed)

        if self.last_epoch == 1 or smoothed < self.loss["best"]:
            self.loss["best"] = smoothed

        return smoothed


class LearningRateFinder:  # pylint:disable=too-many-instance-attributes
    """Learning Rate Finder

    Parameters
    ----------
    enabled
        ``True`` if LRF has been enabled. ``False`` if disabled
    trainer
        The configured and loaded model handler
    selected_lr
        The selected learning rate from the user configuration options
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
                 model_handler: TrainHandler,
                 selected_lr: float,
                 steps: int,
                 strength: T.Literal["default", "aggressive", "extreme"],
                 mode: T.Literal["set", "graph_and_set", "graph_and_exit"],
                 stop_factor: int = 4,
                 beta: float = 0.98) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = "[LearningRateFinder]"
        self._backing_file = os.path.join(model_handler.model_folder,
                                          f"_{model_handler.name}_lrf.ckpt")
        self.is_enabled = self._on_launch(enabled, model_handler, selected_lr)
        """``True`` if LRF has been enabled. ``False`` if disabled"""

        if not self.is_enabled:
            logger.debug("%s Disabled. Exiting early", self._name)
            return

        self._strength = LRStrength[strength.upper()].value
        self._mode = mode
        self._stop_factor = stop_factor

        self._model_handler = model_handler
        self._scheduler = model_handler.optimizer.enable_learning_rate_finder(steps,
                                                                              beta,
                                                                              1e-10,
                                                                              1e-1)

        is_resume = self._scheduler.last_epoch > 0
        logger.info("%s learning rate...", "Resuming" if is_resume else "Finding")
        self._p_bar = tqdm(range(1, self._scheduler.total_steps + 1),
                           desc="Current: N/A      Best: N/A    ",
                           leave=False)
        if is_resume:
            self._update_progress_bar(self._scheduler.last_epoch + 1)

    def _handle_resume(self, model_handler: TrainHandler, selected_lr: float) -> bool:
        """Handle resuming the learning rate finder when model has saved and exited

        Parameters
        ----------
        model_handler
            The object that handles the model and the optimizer
        selected_lr
            The selected learning rate from the user configuration options

        Returns
        -------
        ``True`` if learning rate finder can resume
        """
        if os.path.exists(self._backing_file):
            logger.debug("%s Weights file exists. LRF resumes: '%s'",
                         self._name, self._backing_file)
            sched = model_handler.optimizer.lrf_scheduler
            assert sched is not None
            self._scheduler = sched
            return True

        logger.warning("Resuming Learning Rate Finder, but original weights not found: '%s'",
                       self._backing_file)
        logger.warning("Finder has been cancelled and training will commence at your selected "
                       "learning rate: %s", selected_lr)
        model_handler.optimizer.disable_learning_rate_finder()
        model_handler.optimizer.set_lr(selected_lr)
        return False

    def _on_launch(self, enabled: bool, model_handler: TrainHandler, selected_lr: float) -> bool:
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
        model_handler
            The object that handles the model and the optimizer
        selected_lr
            The selected learning rate from the user configuration options

        Returns
        -------
        ``True`` if the learning rate finder should run in the training loop, otherwise ``False``
        """
        if model_handler.optimizer.lrf_scheduler is not None:  # Only exists when resuming LRF
            return self._handle_resume(model_handler, selected_lr)

        if not enabled:
            logger.debug("%s Disabled", self._name)
            return False

        if model_handler.set_lr_from_finder():
            logger.debug("%s Disabling as LR set from previous Finder", self._name)
            return False

        if model_handler.total_iterations > 0 or model_handler.session_id > 0:
            logger.debug("%s Disabled as not new model", self._name)
            return False
        return True

    def _backup_initial_weights(self) -> None:
        """Back up the initial weights after the first step (when optimizer has been populated)"""
        state_dict = self._model_handler.get_state_dict(with_optimizer=True)
        logger.debug("%s Saving initial weights: '%s'", self._name, self._backing_file)
        torch.save(state_dict, self._backing_file)

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

        smoothed = self._scheduler.track_loss(loss.detach())
        stop_loss = self._stop_factor * self._scheduler.loss["best"]
        if self._scheduler.last_epoch > 1 and smoothed > stop_loss:
            logger.info("Loss has diverged. Exiting early")
            return True

        if self._scheduler.last_epoch == self._scheduler.total_steps:
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
        lrs = self._scheduler.learning_rates[skip_begin:-skip_end]
        losses = T.cast(list[float], self._scheduler.losses[skip_begin:-skip_end])
        plt.plot(lrs, losses, label="Learning Rate")
        best_idx = self._scheduler.losses.index(self._scheduler.loss["best"])
        best_lr = self._scheduler.learning_rates[best_idx]
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
        self._scheduler.losses = [x.item() if isinstance(x, torch.Tensor) else x
                                  for x in self._scheduler.losses]
        best_idx = self._scheduler.losses.index(self._scheduler.loss["best"])
        new_lr = self._scheduler.learning_rates[best_idx] / self._strength
        self._plot_loss()

        if new_lr < 1e-9:
            logger.error("The optimal learning rate could not be found. This is most likely "
                         "because you did not run the finder for enough iterations.")
            logger.debug("%s Removing generated files: %s",
                         self._name, [self._backing_file, self._model_handler.checkpoint_file])
            if os.path.exists(self._model_handler.checkpoint_file):
                os.remove(self._model_handler.checkpoint_file)
            return True

        self._model_handler.handle_lr_finder_completion(new_lr, self._backing_file)
        os.remove(self._backing_file)
        self.is_enabled = False
        return self._mode == "graph_and_exit"

    def _update_progress_bar(self, amount: int | None = None) -> None:
        """Update the description and count of the progress bar for the current iteration

        Parameters
        ----------
        amount
            The amount to iterate the progress bar by. Default: ``None`` (1 step)
        """
        current = self._scheduler.learning_rates[-1]
        best_idx = self._scheduler.losses.index(self._scheduler.loss["best"])
        best = self._scheduler.learning_rates[best_idx] / self._strength
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
            self._backup_initial_weights()

        if self._on_batch_end(loss):
            self._p_bar.close()
            return self._finalize()

        self._update_progress_bar()
        return False


__all__ = get_module_objects(__name__)

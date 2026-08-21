#!/usr/bin/env python3
"""Implements learning rate finder functionality for optimal training configuration.

This module provides tools for automatically determining the best learning rate for a model during
initial exploration. It gradually increases the learning rate from a very small value to a target
level, monitoring loss behavior to identify divergence points and recommend an optimal starting
point for standard training sessions. The finder integrates with PyTorch's scheduler API while
providing smoothed exponential moving average loss tracking for stable analysis.
"""


from __future__ import annotations

from collections import OrderedDict
from enum import Enum
import logging
import os
from datetime import datetime
import typing as T

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .core import TrainingUnit

if T.TYPE_CHECKING:
    from torch.optim import Optimizer
    from lib.model.plugin.train_state import State
    from lib.training.training_loop import TrainStep, Units
    from .core import OptimizerUnit

logger = logging.getLogger(__name__)


class LRStrength(Enum):
    """ Enum values controlling learning rate finder aggressiveness.

    Determines the multiplier applied to optimal learning rates, allowing different strategies
    depending on model stability requirements:

    - **DEFAULT (10x)**: Standard scaling for typical training sessions with moderate
      learning rates (e.g., 1e-4). Used when loss converges smoothly.

    - **AGGRESSIVE (5x)**: For models requiring higher initial learning rates or showing
      slower convergence. Provides finer granularity in the high-LR region of the graph.

    - **EXTREME (2.5x)**: For highly unstable training scenarios where optimal LRs may exceed
      typical ranges. Useful for exploring very large learning rates quickly.

    Attributes
    ----------
    value
        The multiplier used to determine optimal LR thresholds. Lower values indicate more
        aggressive rates with coarser resolution.
    """
    DEFAULT = 10
    AGGRESSIVE = 5
    EXTREME = 2.5


class LRFScheduler(ExponentialLR):
    """ Custom learning rate scheduler with smoothed loss tracking for LRFinder.

    Extends PyTorch's ``ExponentialLR`` to add exponential moving average smoothing of loss values
    during the learning rate finder run. This allows identification of divergence points where
    increasing the LR causes the model to become unstable, which is then used to recommend an
    optimal starting learning rate for production training.

    The scheduler maintains two key smoothed metrics:

    - **smooth_losses**: EMA of recent loss values for stability detection (beta=0.98)
    - **best_loss**: Minimum smooth loss observed during the warmup period

    It tracks both raw and smoothed losses to identify when increasing LR causes divergence,
    which is indicated by ``smoothed > stop_factor * best_loss``.

    Parameters
    ----------
    optimizer
        PyTorch Optimizer instance to control learning rate for. Must have a valid ``lr``
        attribute that will be updated each step.
    start_lr
        Initial learning rate at step 0. Should be very small (e.g., 1e-10). Default: ``1e-10``
    end_lr
        Target maximum learning rate to reach after total_steps iterations. Default: ``1e-1``
    beta
        EMA smoothing factor for loss tracking. Values in [0, 1). Higher = slower decay.
        Default: ``0.98``
    total_steps
        Number of iterations to run the learning rate finder for

    Attributes
    ----------
    smooth_losses
        Running list of exponentially smoothed loss values from each step.
    learning_rates
        History of all computed learning rates after each scheduler.step() call.
    best_loss
        Minimum smoothed loss observed during the warmup period. Used to detect divergence.
    gamma
        Exponential decay factor computed from start_lr/end_lr ratio and total_steps.

    Notes
    -----
    The scheduler implements exponential learning rate scaling:
    .. code-block:: python
        lr = start_lr * (end_lr / start_lr) ** (last_epoch / total_steps)

    Loss smoothing uses the EMA formula:
    .. code-block:: python
        smoothed = beta * avg_loss + (1 - beta) * new_loss

    After final step completes, ``best_loss`` is used to recommend optimal LR by finding
    where loss was minimum before divergence occurred.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 start_lr: float,
                 end_lr: float,
                 beta: float,
                 total_steps: int,
                 last_epoch: int = -1
                 ) -> None:

        self.total_steps = total_steps
        """ The total number of iterations the learning rate finder will run for """
        self.smooth_losses: list[torch.Tensor] = []
        """ Running list of exponentially smoothed loss values from each step. """
        self.learning_rates: list[float] = []
        """ History of all computed learning rates """
        self.best_loss = torch.tensor(float("inf"), dtype=torch.float32)
        """ Minimum smoothed loss observed during the warmup period """

        self._beta = beta
        self._average_loss = torch.tensor(0.0, dtype=torch.float32)

        gamma = (end_lr / start_lr) ** (1.0 / total_steps)
        super().__init__(optimizer, gamma, last_epoch)

    def state_dict(self) -> dict[str, T.Any]:
        """ Serialize the LRFScheduler's state dict for checkpointing

        Returns
        -------
        A dictionary containing all scheduler and finder-specific state variables
        """
        retval = super().state_dict()
        retval["beta"] = self._beta
        retval["total_steps"] = self.total_steps
        retval["smooth_losses"] = self.smooth_losses  # TODO name change for state_dict
        retval["learning_rates"] = self.learning_rates
        retval["average_loss"] = self._average_loss
        retval["best_loss"] = self.best_loss
        return retval

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """ Restore scheduler's internal state from a serialized dictionary representation

        Parameters
        ----------
        state_dict
            A dictionary containing all state variables from ``state_dict()``
        """
        # TODO do avg and best need to come from state_dict or be re-calculated?
        self._beta = state_dict.pop("beta")
        self.total_steps = state_dict.pop("total_steps")
        self.smooth_losses = state_dict.pop("smooth_losses")  # TODO name change for state_dict
        self.learning_rates = state_dict.pop("learning_rates")
        self._average_loss = state_dict.pop("average_loss")
        self.best_loss = state_dict.pop("best_loss")
        super().load_state_dict(state_dict)

    def step(self, epoch: int | None = None, loss: torch.Tensor | None = None) -> None:
        """ Advance scheduler by one iteration and optionally update loss smoothing.

        Calls parent class to advance learning rate calculation, then updates smoothed loss
        tracking if a new loss value is provided and we're past the initial warmup phase.

        Parameters
        ----------
        epoch
            The current training epoch (unused for LRFinder - always increments internally)
        loss
            Optional current loss value to use for smoothing calculations. Required for
            computing smoothed losses during active learning rate finder runs. Default: ``None``

        Notes
        -----
        This method performs three actions in sequence:
        1. **Parent step**: Calls super().step() to advance the exponential LR schedule
        2. **Loss smoothing**: If loss is provided and we're past epoch 0, computes EMA of losses

            .. code-block:: python
                smoothed = beta * avg_loss + (1 - beta) * loss

        3. **Best loss tracking**: Updates minimum observed smooth loss for divergence detection
        """
        super().step(epoch=epoch)
        if self.last_epoch <= 0 or loss is None:
            return

        self.learning_rates.append(T.cast(float, self.get_last_lr()[0]))
        self._average_loss = (self._beta * self._average_loss) + ((1 - self._beta) * loss)

        smoothed: torch.Tensor = self._average_loss / (1 - (self._beta ** self.last_epoch))
        self.smooth_losses.append(smoothed)
        self.best_loss = min(self.best_loss, smoothed)


class LearningRateFinder:
    """ Manages learning rate finder execution and early termination logic.

    Orchestrates the learning rate search by running the scheduler for configured iterations,
    monitoring loss behavior to detect divergence points, and determining when to stop early.
    Implements exponential decay tracking of minimum loss values to identify optimal LR ranges.

    Parameters
    ----------
    scheduler
        The ``LRFScheduler`` instance controlling LR progression through warmup period. Must be
        initialized with valid parameters before this finder can operate.
    strength
        Aggressiveness level from ``LRStrength`` enum that controls the final learning rate.
        Lower values (aggressive/extreme) provide finer granularity for high-LR regions.

    Notes
    -----
    This class integrates with the training loop via its ``step()`` method which accepts loss
    values and advances the scheduler internally. It handles three termination conditions:

    1. **Early exit on NaN**: If loss becomes NaN, immediately finalizes as no need to proceed
    2. **Divergence detection**: Stops when smoothed loss > stop_factor * best_loss (default: 4x)
    3. **Complete ramp-up**: After reaching total_steps configured in scheduler
    """
    def __init__(self,
                 scheduler: LRFScheduler,
                 strength: T.Literal["default", "aggressive", "extreme"],
                 stop_factor: int = 4) -> None:
        logger.debug(parse_class_init(locals()))
        self._scheduler = scheduler
        self._strength = LRStrength[strength.upper()].value
        self._stop_factor = stop_factor

        self._p_bar = tqdm(range(1, scheduler.total_steps + 1),
                           desc="Current: N/A      Best: N/A    ",
                           leave=False)
        self._optimal_learning_rate: float | None = None

    @property
    def optimal_learning_rate(self) -> float:
        """ The recommended optimal learning rate for production training """
        assert self._optimal_learning_rate is not None, "LRFinder has not been run"
        return self._optimal_learning_rate

    def _get_best_learning_rate(self) -> float:
        """ Determine the optimal learning rate by finding minimum loss point in smoothed curve

        Returns
        -------
        The best learning rate divided by strength multiplier for use as initial training LR.
        """
        best_idx = self._scheduler.smooth_losses.index(self._scheduler.best_loss)
        return self._scheduler.learning_rates[best_idx] / self._strength

    def update_progress_bar(self, amount: int | None = None) -> None:
        """ Update the progress bar with current metrics and status.

        Parameters
        ----------
        amount
            Steps to advance the progress bar by. If ``None``, advances by 1. Default: ``None``
        """
        current = self._scheduler.learning_rates[-1]
        best = self._get_best_learning_rate()
        self._p_bar.update(1 if amount is None else amount)
        self._p_bar.set_description(f"Current: {current:.1e}  Best: {best:.1e}")

    def _finalize(self) -> bool:
        """ Complete the learning rate finder run and finalize results.

        Closes the progress bar, clears console output line, calculates final optimal
        learning rate, and returns success status for caller to handle cleanup logic.

        Returns
        -------
        bool
            Always ``True`` - signals that finding process has successfully completed.
        """
        self._p_bar.close()
        print("\x1b[2K", end="\r")  # Clear line
        self._optimal_learning_rate = self._get_best_learning_rate()
        return True

    def step(self, loss: torch.Tensor) -> bool:
        """ Advance learning rate finder by one step and check for termination conditions.

        Executes a single iteration of the learning rate search by advancing the scheduler,
        checking for divergence or NaN loss, and updating progress display until completion.
        Returns when early exit occurs due to convergence issues or final step reached.

        Parameters
        ----------
        loss
            Current batch loss value from training loop. Used for EMA smoothing and divergence
            detection calculations.

        Returns
        -------
        bool
            ``True`` when finder has finished (early exit, completion, or error), ``False``
            otherwise to continue with next iteration.

        Notes
        -----
        This method performs three key checks in sequence:
        1. **NaN Detection**: Immediately terminates if loss contains NaN values, indicating
           training instability that prevents meaningful LR analysis.
        2. **Scheduler Step**: Advances the ``LRFScheduler`` by calling its internal step()
           to update learning rate and compute smoothed losses.
        3. **Termination Conditions**: Checks for three exit scenarios:
            - *Divergence*: When smoothed loss exceeds ``stop_factor * best_loss`` (default: 4x)
            - *Completion*: After reaching total configured steps without divergence
            - *NaN Loss*: If loss becomes invalid at any point

        Each termination triggers finalization which computes optimal learning rate and
        closes resources. The method returns ``True`` only when a termination condition is met.
        """
        if torch.isnan(loss):
            logger.info("[LearningRateFinder] Loss has NaN'd. Exiting early")
            return self._finalize()

        self._scheduler.step(loss=loss)

        smoothed = self._scheduler.smooth_losses[-1]
        stop_loss = self._stop_factor * self._scheduler.best_loss

        if self._scheduler.last_epoch > 1 and smoothed > stop_loss:
            logger.info("[LearningRateFinder] Loss has diverged. Exiting early")
            return self._finalize()

        if self._scheduler.last_epoch == self._scheduler.total_steps:
            logger.debug("[LearningRateFinder] Reached final step. Exiting")
            return self._finalize()

        self.update_progress_bar()
        return False


def plot_loss(filename: str,
              learning_rates: list[float],
              losses: list[float],
              best_loss: float,
              skip_begin: int = 10,
              skip_end: int = 1) -> None:
    """ Generate and save a Learning Rate Finder visualization graph.

    Creates a log-scaled plot showing the relationship between learning rates and smoothed
    losses during the finder run. Marks optimal learning rate thresholds for different
    aggressiveness levels to guide training configuration decisions

    Parameters
    ----------
    filename
        Path where the generated PNG image will be saved. File extension should be .png
    learning_rates
        Complete history of learning rates from scheduler.step() calls throughout finder run
    losses
        Corresponding smoothed loss values for each learning rate step (EMA computed)
    best_loss
        Minimum observed smoothed loss value during the warmup/divergence analysis period.
        Used to identify optimal operating point on the curve
    skip_begin, optional
        Number of initial points to exclude from plot. Removes early noisy steps where
        learning rate is too small and model hasn't adapted yet. Default: 10
    skip_end, optional
        Number of final points to exclude from plot. Removes the last few steps which may
        show divergence behavior that isn't useful for interpretation. Default: 1

    Notes
    -----
    The plot displays three colored markers indicating optimal learning rate thresholds:

    - **Green (default)**: Best LR / 10x - Suitable for stable training with moderate rates
    - **Yellow (aggressive)**: Best LR / 5x - For models needing higher initial rates
    - **Red (extreme)**: Best LR / 2.5x - For unstable scenarios requiring careful exploration

    The X-axis uses log scale to span the wide range of learning rates encountered during
    the finder run, while Y-axis shows loss values decreasing then increasing at divergence.

    Examples
    --------
    >>> plot_loss("lrf_graph.png", lrs, losses, best)

    Saves a PNG showing LR vs Loss relationship with recommended thresholds marked.
    """
    matplotlib.use("Agg")
    lrs = learning_rates[skip_begin:-skip_end]
    all_losses = losses
    losses = all_losses[skip_begin:-skip_end]

    plt.plot(lrs, losses, label="Learning Rate")

    best_idx = all_losses.index(best_loss)
    best_lr = learning_rates[best_idx]
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

    logger.info("[LearningRateFinder] Saving graph to: '%s'", filename)
    plt.savefig(filename)


class LRFState:
    def __init__(self,  # pylint:disable=too-many-instance-attributes
                 loop: TrainStep,
                 scheduler: LRFScheduler,
                 finder: LearningRateFinder,
                 mode: T.Literal["set", "graph_and_set", "graph_and_exit"],
                 start_lr: float) -> None:
        logger.debug(parse_class_init(locals()))
        self._log_name = self.__class__.__name__

        self._scheduler = scheduler
        self._lrf = finder
        self._mode = mode
        self._start_lr = start_lr

        self._model = loop.model
        self._optimizer = loop.optimizer_unit
        self._events = loop.events
        self._current_loss = loop.current_loss
        self._checkpoint_file = self._model.io.checkpoint_path

        folder, fname = os.path.split(os.path.splitext(self._model.io.checkpoint_path)[0])
        self._backing_file = os.path.join(folder, f"_{fname}_lrf.ckpt")

    def _backup_initial_weights(self) -> None:
        state_dict = OrderedDict({"model": self._model.plugin.state_dict(),
                                  "optimizer": self._optimizer.state_dict()})
        logger.debug("%s Saving initial weights: '%s'", self._log_name, self._backing_file)
        torch.save(state_dict, self._backing_file)

    def on_start(self):
        self._optimizer.set_lr(self._start_lr)
        self._backup_initial_weights()
        self._model.state.set_pre_training()

    def _save_graph(self) -> None:
        if self._mode not in ("graph_and_set", "graph_and_exit"):
            return

        assert self._scheduler is not None
        now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        graph_file = f"{os.path.splitext(self._checkpoint_file)[0]}_lr_finder_{now}.png"
        plot_loss(graph_file,
                  learning_rates=self._scheduler.learning_rates,
                  losses=[x.item() for x in self._scheduler.smooth_losses],
                  best_loss=self._scheduler.best_loss.item())

    def _validate_result(self, new_lr: float) -> bool:
        if new_lr <= self._start_lr:
            logger.error("The optimal learning rate could not be found. This is most likely "
                         "because you did not run the finder for enough iterations.")
            for fname in [self._backing_file, self._checkpoint_file]:
                if os.path.exists(self._checkpoint_file):
                    logger.debug("%s Removing generated file: %s",
                                 self._log_name, self._checkpoint_file)
                    os.remove(fname)
            return False
        return True

    def _set_learning_rate(self, new_lr: float) -> None:
        logger.debug("%s Storing result in state file: %s", self._log_name, new_lr)
        if self._mode == "graph_and_exit":
            logger.info("[LearningRateFinder] Setting learning rate to: %s", f"{new_lr:.1e}")
        self._model.state.session_config["learning_rate"] = new_lr
        self._model.state.lr_finder = new_lr
        self._optimizer.set_lr(new_lr)

    def _restore_initial_weights(self) -> None:
        logger.debug("%s Restoring model weights from: '%s'", self._log_name, self._backing_file)
        original_weights = torch.load(self._backing_file)
        self._model.plugin.load_state_dict(original_weights["model"])
        self._optimizer.load_state_dict(original_weights["optimizer"])

    def _save_original_model(self) -> None:
        if self._mode == "graph_and_exit":
            self._events.exit.set()  # Let main loop handle saving
            return

        logger.info("[LearningRateFinder] Restoring original model state")
        fresh_state = self._model.state_dict()
        fresh_state["optimizer"] = self._optimizer.state_dict()
        self._model.io.save(fresh_state)

    def _clean_up(self):
        del self._scheduler
        self._scheduler = None
        logger.debug("%s Disabled learning rate scheduler", self._log_name)
        os.remove(self._backing_file)

    def _finalize(self) -> None:
        self._save_graph()

        new_lr = self._lrf.optimal_learning_rate
        if not self._validate_result(new_lr):
            self._events.exit.set()
            return

        self._set_learning_rate(new_lr)
        self._restore_initial_weights()
        self._clean_up()
        self._model.state.set_training()
        self._save_original_model()

    def step(self) -> bool:
        loss = T.cast("torch.Tensor", sum(x.total for x in self._current_loss))
        if self._lrf.step(loss):
            self._finalize()
            return True
        return False

    def resume(self) -> None:
        assert self._scheduler is not None
        self._lrf.update_progress_bar(len(self._scheduler.smooth_losses))


class LRFinderUnit(TrainingUnit):
    def __init__(self,
                 steps: int = -1,
                 strength: T.Literal["default", "aggressive", "extreme"] | None = None,
                 mode: T.Literal["set", "graph_and_set", "graph_and_exit"] | None = None,
                 stop_factor: int = 4,
                 start_lr: float = 1e-10,
                 end_lr: float = 1e-1,
                 beta: float = 0.98) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._scheduler_kwargs = {"start_lr": start_lr,
                                  "end_lr": end_lr,
                                  "beta": beta,
                                  "total_steps": steps}
        self._lrf_state_kwargs = {"mode": mode, "start_lr": start_lr}
        self._lrf_kwargs = {"strength": strength, "stop_factor": stop_factor}

        self._lrf_state: LRFState | None = None  # set in on_start if required
        self._scheduler: LRFScheduler | None = None  # set in on_start if required

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"steps={self._scheduler_kwargs['total_steps']!r}, "
                f"strength={self._lrf_kwargs['strength']!r}, "
                f"mode={self._lrf_state_kwargs['mode']!r}, "
                f"stop_factor={self._lrf_kwargs['stop_factor']!r}, "
                f"start_lr={self._scheduler_kwargs['start_lr']!r}, "
                f"end_lr={self._scheduler_kwargs['end_lr']!r}, "
                f"beta={self._scheduler_kwargs['beta']!r})")

    # ## ON START ##
    def _kwargs_from_config(self, config: dict[str, T.Any]) -> None:
        if self._scheduler_kwargs["total_steps"] == -1:
            self._scheduler_kwargs["total_steps"] = config["lr_finder_iterations"]
            logger.debug("%s Set steps from config: %s",
                         self.log_name, self._scheduler_kwargs["total_steps"])

        if self._lrf_kwargs["strength"] is None:
            self._lrf_kwargs["strength"] = config["lr_finder_strength"]
            logger.debug("%s Set strength from config: '%s'",
                         self.log_name, self._lrf_kwargs["strength"])

        if self._lrf_state_kwargs["mode"] is None:
            self._lrf_state_kwargs["mode"] = config["lr_finder_mode"]
            logger.debug("%s Set mode from config: '%s'",
                         self.log_name, self._lrf_state_kwargs["mode"])

    def _set_learning_rate_from_lrf(self,
                                    state: State,
                                    optimizer: OptimizerUnit,
                                    units: Units) -> None:
        new_lr = state.lr_finder
        assert new_lr > 0, "LRF information has not been stored"
        logger.info("[LearningRateFinder] Setting learning rate to: %s", f"{new_lr:.1e}")

        state.session_config["learning_rate"] = new_lr
        optimizer.set_lr(new_lr)

        logger.debug("%s removing self from steppers", self.log_name)
        units.stages_optional["step"].remove(self)

    def _setup_lr_finder(self, loop: TrainStep) -> tuple[LRFScheduler, LRFState]:
        self._kwargs_from_config(loop.model.state.config)
        scheduler = LRFScheduler(loop.optimizer_unit.optimizer, **self._scheduler_kwargs)
        lrf = LearningRateFinder(scheduler=scheduler, **self._lrf_kwargs)
        lrf_state = LRFState(loop, scheduler, lrf, **self._lrf_state_kwargs)
        logger.info("[LearningRateFinder] start: %s, end: %s, steps: %s",
                    self._scheduler_kwargs["start_lr"],
                    self._scheduler_kwargs["end_lr"],
                    self._scheduler_kwargs["total_steps"])
        lrf_state.on_start()
        return scheduler, lrf_state

    def on_start(self, loop: TrainStep) -> None:
        is_resume = (loop.model.state.session_id == 1 and
                     loop.model.io.file_exists and
                     loop.model.state.iterations < 0)
        if is_resume:
            logger.debug("%s Resuming LRF", self.log_name)

        if not is_resume and (loop.model.state.session_id != 1 or loop.model.io.file_exists):
            self._set_learning_rate_from_lrf(loop.model.state, loop.optimizer_unit, loop.units)
            return

        self._scheduler, self._lrf_state = self._setup_lr_finder(loop)

    def step(self, iteration: int) -> None:
        if iteration > 0:
            return  # LRF finder has run and we are now training
        assert self._lrf_state is not None
        if self._lrf_state.step():
            del self._scheduler
            self._scheduler = None

    def state_dict(self) -> dict[str, T.Any]:
        if self._scheduler is None:
            return {}
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        # Learning Rate Finder resume
        if self._scheduler is None:  # TODO we need to catch this if we are resuming LRF
            logger.debug("%s No LRF scheduler. Not loading state_dict", self.log_name)
            return
        assert self._lrf_state is not None
        logger.info("[LearningRateFinder] Resuming")
        self._scheduler.load_state_dict(state_dict)
        self._lrf_state.resume()


__all__ = get_module_objects(__name__)

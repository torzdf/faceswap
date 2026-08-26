#! /usr/bin/env python3
""" Handles learning rate finding to determine optimal training parameters

This optional module provides functionality for performing learning rate finder (LRF) analysis to
help users identify optimal learning rates before beginning main model training. It uses an
exponential decay schedule with exponential moving average loss smoothing to find the point where
the model's loss has stabilized, indicating a good starting learning rate for subsequent training.

The module includes:
- LRFScheduler: Custom PyTorch scheduler implementing exponential decay learning rate schedule
- LearningRateFinder: Controls the LRF sweep and handles early stopping based on divergence/NaN
- plot_loss: Generates visualization of LRF results showing loss vs learning rate curves
- LRFState: Manages model state preservation/restore during LRF process
- LRFinderUnit: TrainingUnit integration for LRF lifecycle management within training loop

The finder operates by starting at a very low learning rate (1e-10) and exponentially increasing
to a high value (1e-1), tracking the loss curve. The optimal point is where loss decreases at the
fastest rate. Initial model weights are backed up before LRF begins and restored afterwards to
maintain training continuity
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

from lib.training.units.core import TrainingUnit

if T.TYPE_CHECKING:
    from torch.optim import Optimizer
    from lib.model.plugin import State
    from lib.training.training_loop import TrainStep, Units
    from lib.training.units.core import OptimizerUnit

logger = logging.getLogger(__name__)


class LRStrength(Enum):
    """ Enum values controlling learning rate finder aggressiveness

    Determines the multiplier applied to the final optimal learning rate, allowing different
    strategies depending on model stability requirements:
      - **DEFAULT (10x)**: Standard scaling for typical training sessions with moderate learning
      rates. Used when loss converges smoothly.
      - **AGGRESSIVE (5x)**: For models requiring higher initial learning rates or showing slower
      convergence. Provides finer granularity in the high-LR region of the graph.
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
    """ Custom learning rate scheduler for LRF with loss tracking and smoothing

    This scheduler extends PyTorch's ExponentialLR to implement the exponential decay schedule used
    during learning rate finding sweeps. It tracks both learning rates at each step and computes an
    exponentially weighted moving average of losses, allowing identification of the optimal point
    where loss is decreasing fastest. The gamma parameter is computed from start/end LR values
    over total steps to ensure smooth exponential progression throughout the sweep

    Parameters
    ----------
    optimizer
        The Torch optimizer whose learning rate will be adjusted at each step during the LRF sweep
    start_lr
        Initial learning rate for the sweep, typically 1e-10 (very small to ensure stable starting
        point)
    end_lr
        Final learning rate after complete sweep, typically 1e-1 (large value ensuring divergence
        at end)
    beta
        Smoothing coefficient for exponential moving average of losses. Higher values give more
        weight
        to recent losses (default: 0.98).
    total_steps
        Number of steps over which to perform the LRF sweep. Determines how many learning rate
        changes occur
    last_epoch, optional
        The index of the last epoch for scheduler state management. Default: ``-1``

    Attributes
    ----------
    total_steps
        Total number of LRF sweep iterations planned
    smooth_losses
        Exponentially smoothed loss values computed at each step (exponential moving average)
    learning_rates
        Learning rate value applied at each step during the sweep
    best_loss
        Best (minimum) smoothed loss encountered so far during the sweep, used for early stopping
        decision

    Notes
    -----
    The learning rate at each step follows the formula:
    lr_i = start_lr * (end_lr/start_lr)^(i/total_steps).
    This ensures smooth exponential progression from very small starting value to large ending
    value
    """
    def __init__(self,
                 optimizer: Optimizer,
                 start_lr: float,
                 end_lr: float,
                 beta: float,
                 total_steps: int,
                 last_epoch: int = -1
                 ) -> None:
        logger.debug(parse_class_init(locals()))

        self.total_steps = total_steps
        """ Total number of LRF sweep iterations planned """
        self.smooth_losses: list[torch.Tensor] = []
        """ Exponentially smoothed loss values computed at each step """
        self.learning_rates: list[float] = []
        """ Learning rate value applied at each step during the sweep """
        self.best_loss = torch.tensor(float("inf"), dtype=torch.float32)
        """ Best (minimum) smoothed loss encountered so far during the sweep """
        self._beta = beta
        self._average_loss = torch.tensor(0.0, dtype=torch.float32)

        gamma = (end_lr / start_lr) ** (1.0 / total_steps)
        super().__init__(optimizer, gamma, last_epoch)

    def state_dict(self) -> dict[str, T.Any]:
        """ Return scheduler state dictionary for checkpointing

        Returns
        -------
        Dictionary containing all custom LRF attributes plus standard ExponentialLR state:
          - ``beta``: The smoothing coefficient used in EMA computation
          - ``total_steps``: Total planned sweep iterations
          - ``smooth_losses``: List of smoothed loss values at each step
          - ``learning_rates``: Learning rates applied at each step during sweep
          - ``average_loss``: Current exponentially weighted average of losses
          - ``best_loss``: Best (minimum) smoothed loss encountered so far
          - Standard PyTorch ExponentialLR state_dict items
        """
        retval = super().state_dict()
        retval["beta"] = self._beta
        retval["total_steps"] = self.total_steps
        retval["smooth_losses"] = self.smooth_losses
        retval["learning_rates"] = self.learning_rates
        retval["average_loss"] = self._average_loss
        retval["best_loss"] = self.best_loss
        return retval

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """ Restore scheduler from saved state dictionary

        Parameters
        ----------
        state_dict
            Dictionary containing all LRF attributes loaded from previous checkpoint
        """
        self._beta = state_dict.pop("beta")
        self.total_steps = state_dict.pop("total_steps")
        self.smooth_losses = state_dict.pop("smooth_losses")
        self.learning_rates = state_dict.pop("learning_rates")
        self._average_loss = state_dict.pop("average_loss")
        self.best_loss = state_dict.pop("best_loss")
        super().load_state_dict(state_dict)

    def step(self, epoch: int | None = None, loss: torch.Tensor | None = None) -> None:
        """ Advance learning rate and track loss smoothing

        Calls parent class to update internal ExponentialLR state (base_lrs, last_epoch). Loss
        should always be provided after the first initialization iteration. This method also
        updates the exponential moving average of losses and appends both the current learning rate
        and smoothed loss to their respective tracking lists. This enables identification of the
        optimal point where loss has stabilized.

        Parameters
        ----------
        epoch, optional
            Epoch number for scheduler state. Default: ``None``
        loss, optional
            Loss value from current batch to compute smoothed average. Should always be provided.
            Default: ``None``
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
    """ Controller class for performing learning rate sweep operations

    This class manages the complete LRF sweep process including progress bar display, early
    stopping checks (NaN loss or divergence), and computation of optimal learning rate from the
    loss curve minimum point. It uses a tqdm progress bar that updates after each step with
    current/best learning rate values displayed in the description

    Parameters
    ----------
    scheduler
        The custom scheduler instance whose learning rates will be advanced during the sweep
    strength
        Aggressiveness level for finding optimal LR, determining how to divide best_loss result
    stop_factor, optional
        Factor by which smoothed loss must exceed best_loss before early stopping due to
        divergence. Default: ``4`` (stops when loss > 4x best value)
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
        """ The computed optimal learning rate. Raises AssertionError if LRF has not been run """
        assert self._optimal_learning_rate is not None, "LRFinder has not been run"
        return self._optimal_learning_rate

    def _get_best_learning_rate(self) -> float:
        """ Compute optimal LR from loss curve minimum point

        Returns
        -------
        float
            The computed optimal learning rate after dividing best_loss LR by strength factor
        """
        best_idx = self._scheduler.smooth_losses.index(self._scheduler.best_loss)
        return self._scheduler.learning_rates[best_idx] / self._strength

    def update_progress_bar(self, amount: int | None = None) -> None:
        """ Update progress bar with current and best learning rates

        Parameters
        ----------
        amount, optional
            Number of progress steps to advance. If None advances by 1 step. Defaul: ``None``
        """
        current = self._scheduler.learning_rates[-1]
        best = self._get_best_learning_rate()
        self._p_bar.update(1 if amount is None else amount)
        self._p_bar.set_description(f"Current: {current:.1e}  Best: {best:.1e}")

    def _finalize(self) -> bool:
        """ Finalize the LRF sweep and compute optimal learning rate

        Returns
        -------
        bool
            Always returns True to signal completion (convention from step() return pattern)
        """
        self._p_bar.close()
        print("\x1b[2K", end="\r")  # Clear line
        self._optimal_learning_rate = self._get_best_learning_rate()
        return True

    def step(self, loss: torch.Tensor) -> bool:
        """ Perform one LRF iteration and check for stopping conditions

        Advances the scheduler's learning rate, updates loss smoothing if this is not the initial
        step (last_epoch > 0), checks for early stopping triggers (NaN or divergence), and updates
        progress bar. Returns True when sweep should end

        Parameters
        ----------
        loss
            Loss value from current batch to track in exponential moving average

        Returns
        -------
        ``True`` if LRF has completed (either normally or via early stop), ``False`` to continue

        Notes
        -----
        Early stopping occurs when:
        1. Loss is NaN (model unstable at current LR, too aggressive)
        2. Smoothed loss > stop_factor x best_loss (loss diverged)
        3. Reached final scheduled step (total_steps iterations completed)
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
    """ Generate and save LRF results visualization as PNG file

    Creates a log-scale plot showing learning rate on x-axis (log scale) versus loss values on y-
    axis. The curve helps identify the optimal point where loss has stabilized at minimum. Markers
    are added indicating the aggressive/defaut/extreme strength levels for reference, with color-
    coded circles showing where each aggressiveness would suggest changing LR

    Parameters
    ----------
    filename
        Output path for saving the generated PNG visualization file
    learning_rates
        Learning rates encountered during LRF sweep (will be sliced to exclude first/last points)
    losses
        Corresponding loss values at each learning rate point (also sliced consistently)
    best_loss
        The minimum loss value on the curve, used for identifying optimal LR marker position
    skip_begin
        Number of initial points to exclude from plotting (removes unstable early sweep data).
        Default: ``10``
    skip_end : int, optional
        Number of final points to exclude (removes noisy end-of-sweep divergence region).
        Default: ``1``
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

    logger.info("[LearningRateFinder] Graph saved: '%s'", filename)
    plt.savefig(filename)


class LRFState:  # pylint:disable=too-many-instance-attributes
    """ State manager for coordinating model backup/restore during LRF process.

    This class handles all aspects of the learning rate finder lifecycle including initial
    weight backup before sweep, scheduler creation and configuration, results storage in state
    file after completion, and weight restoration to allow uninterrupted training continuation.
    It manages events for exit signaling and mode handling (set LR only vs save graph then exit).

    Parameters
    ----------
    loop
        Training step object providing access to model, optimizer, current loss tracker, events
        system, etc.
    scheduler
        The learning rate scheduler instance used during the sweep operation
    finder
        Controller for managing LRF sweep execution and stopping conditions
    mode
        Operation mode: "set" only sets the learning rate and continues training. "graph_and_set"
        saves graph then sets LR and continues training. "graph_and_exit" saves graph to file and
        exits the training loop
    start_lr
        Starting learning rate for the LRF sweep (typically 1e-10 for stability)
    """
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
        """ Save model and optimizer states to backup file before LRF sweep """
        state_dict = OrderedDict({"model": self._model.plugin.state_dict(),
                                  "optimizer": self._optimizer.state_dict()})
        logger.debug("%s Saving initial weights: '%s'", self._log_name, self._backing_file)
        torch.save(state_dict, self._backing_file)

    def on_load(self) -> None:
        """ Initialize LRF by setting start LR, backing up weights, and entering pre-training mode.

        Sets the optimizer's learning rate to _start_lr (typically 1e-10), saves initial model/
        optimizer states, then switches model state from normal training to pre_training flag in
        state dictionary.
       """
        self._optimizer.set_lr(self._start_lr)
        self._backup_initial_weights()
        self._model.state.set_pre_training()

    def _save_graph(self) -> None:
        """ Generate LRF results visualization if mode requires graph output """
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
        """ Validate that optimal LR is greater than starting value

        Parameters
        ----------
        new_lr
            The optimal learning rate computed from LRF sweep results

        Returns
        -------
        ``True`` if `new_lr` > `start_lr` (valid result), ``False`` if `new_lr` <= `start_lr`
        """
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

    def _restore_initial_weights(self) -> None:
        """ Load saved backup weights and restore model to pre-LRF state """
        logger.debug("%s Restoring model weights from: '%s'", self._log_name, self._backing_file)
        original_weights = torch.load(self._backing_file)
        self._model.plugin.load_state_dict(original_weights["model"])
        self._optimizer.load_state_dict(original_weights["optimizer"])

    def _set_learning_rate(self, new_lr: float) -> None:
        """ Store optimal learning rate in state and update optimizer

        Parameters
        ----------
        new_lr
            The optimal learning rate computed by LRF to replace current value
        """
        logger.debug("%s Storing result in state file: %s", self._log_name, new_lr)
        if self._mode != "graph_and_exit":
            logger.info("[LearningRateFinder] Setting learning rate to: %s", f"{new_lr:.1e}")
        self._model.state.session_config["learning_rate"] = new_lr
        self._model.state.lr_finder = new_lr
        self._optimizer.set_lr(new_lr)

    def _save_original_model(self) -> None:
        """ Save fresh initial model state dict after the LRF run """
        if self._mode == "graph_and_exit":
            self._events.exit.set()  # Let main loop handle saving
            return

        logger.debug("[LearningRateFinder] Restoring original model state")
        fresh_state = self._model.state_dict()
        fresh_state["optimizer"] = self._optimizer.state_dict()
        fname = self._model.checkpoint_path
        logger.debug("%s Saving original checkpoint: '%s'", self._log_name, fname)
        torch.save(fresh_state, fname)

    def _clean_up(self):
        """ Delete scheduler instance and remove backup file to clean up resources """
        del self._scheduler
        self._scheduler = None
        logger.debug("%s Disabled learning rate scheduler", self._log_name)
        os.remove(self._backing_file)

    def _finalize(self) -> None:
        """ Complete LRF process by saving graph, validating result, setting LR and cleaning up """
        self._save_graph()

        new_lr = self._lrf.optimal_learning_rate
        if not self._validate_result(new_lr):
            self._events.exit.set()
            return

        self._restore_initial_weights()
        self._set_learning_rate(new_lr)
        self._clean_up()
        self._model.state.set_training()
        self._save_original_model()

    def step(self) -> bool:
        """ Perform one LRF sweep iteration and check if completed

        Computes total loss from current_loss tracker, passes it to finder.step() which advances
        scheduler, checks stopping conditions (divergence/NaN/end), updates progress bar. Returns
        True when the LRF has finished, False otherwise to continue loop.

        Returns
        -------
        True if LRF completed and should exit sweep, False to continue iterating

        Notes
        -----
        When iteration > 0 returns early without doing work since first call already set up pre-
        training mode. Subsequent calls are for main training loop which should skip LRF steps.
        """

        loss = T.cast("torch.Tensor", sum(x.total for x in self._current_loss))
        if self._lrf.step(loss):
            self._finalize()
            return True
        return False

    def resume(self) -> None:
        """ Restore progress bar state when resuming interrupted LRF from checkpoint

        Assumes scheduler is already loaded (not None), then jumps the progress bar to current
        position by updating it with len(smooth_losses) steps. This makes the visual output match
        where sweep left off for user awareness, even though actual computation will continue
        fresh.

        Raises
        ------
        AssertionError
            If scheduler is None (shouldn't occur in resume context)

        Notes
        -----
        Only called when resuming from checkpoint - during normal LRF this method is never invoked.
        The progress bar update doesn't restore lost computation results, just visual state.
        """
        assert self._scheduler is not None
        self._lrf.update_progress_bar(len(self._scheduler.smooth_losses))


class LRFinderUnit(TrainingUnit):
    """ Learning rate finder unit integrating with training loop lifecycle.

    This optional TrainingUnit manages the complete learning rate finder lifecycle within the
    training system: initialization from config values, checkpoint loading on resume, sweep
    execution via LRFState step() calls, result application after completion, and proper cleanup
    when training continues normally. It wraps the core LRF components (LRFScheduler,
    LearningRateFinder) with configuration parsing and state management for seamless integration.

    The unit automatically detects if it should resume from checkpoint or run new sweep based on
    session_id and file existence checks. If resuming, loads stored results and sets LR directly
    without running finder again. Otherwise performs complete sweep using configured parameters.

    Parameters
    ----------
    steps
        Number of iterations to perform during LRF sweep. Default: ``-1`` (load from config)
    strength, optional
        Determines the multiplier applied to the final optimal learning rate. Default: ``None``
        (load from config)
    mode, optional
        Operation mode determining what happens after LRF completes. "set" only changes LR,
        "graph_and_set" saves graph then sets, "graph_and_exit" saves graph for manual review
        instead of auto-setting LR. Default: ``None`` (load from config)
    stop_factor, optional
        Divergence threshold multiplier - sweep stops early when loss exceeds this x best_loss.
        Default: ``4`` (stops at 4x best value to prevent runaway training)
    start_lr, optional
        Starting learning rate for LRF exponential decay schedule. Should be low enough that model
        is stable during initial sweep iterations. Default: ``1e-10`` (very small)
    end_lr, optional
        Target learning rate after complete sweep - determines where exponential curve ends.
        Default: ``1e-1`` (large value ensuring divergence at end for proper curve shape)
    beta, optional
        Smoothing coefficient for exponential moving average of losses during sweep. Higher values
        weight recent losses more heavily. Default: ``0.98`` (standard EMA decay rate)

    Notes
    -----
    The unit integrates with TrainingUnit lifecycle: on_load() initializes components, step()
    performs sweep iterations, and state_dict()/load_state_dict() handle checkpoint persistence.
    When LRF completes successfully (step() returns True), the unit removes itself from steppers so
    main training can continue normally without further LRF interference.
    """
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

        self._lrf_state: LRFState | None = None  # set in on_load if required
        self._scheduler: LRFScheduler | None = None  # set in on_load if required
        self._units: Units  # set in on_load

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
        """ Parse configuration dictionary to fill in default parameter values

        Parameters
        ----------
        config
            Configuration dictionary containing keys: "lr_finder_iterations", "lr_finder_strength",
            "lr_finder_mode" for parameter resolution
        """
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
                                    optimizer: OptimizerUnit) -> None:
        """ Apply stored optimal LR

        Parameters
        ----------
        state
            Model's state object containing lr_finder attribute with stored optimal rate
        optimizer
            Reference to optimizer unit whose learning_rate method must be called

        Raises
        ------
        AssertionError
            If new_lr <= 0 (shouldn't occur if LRF stored valid result before saving state)
        """
        new_lr = state.lr_finder
        assert new_lr > 0, "LRF information has not been stored"
        logger.info("[LearningRateFinder] Setting learning rate to: %s", f"{new_lr:.1e}")

        state.session_config["learning_rate"] = new_lr
        optimizer.set_lr(new_lr)

    def _setup_lr_finder(self, loop: TrainStep) -> tuple[LRFScheduler, LRFState]:
        """ Create and initialize LRFScheduler and LRFState instances with full configuration

        Parameters
        ----------
        loop
            Training step providing access to optimizer, events, current_loss for sched/state init

        Returns
        -------
        scheduler
            The initialized learning rate scheduler
        state
            The initialized learning rate state
        """
        self._kwargs_from_config(loop.model.state.config)
        scheduler = LRFScheduler(loop.optimizer_unit.optimizer, **self._scheduler_kwargs)
        lrf = LearningRateFinder(scheduler=scheduler, **self._lrf_kwargs)
        lrf_state = LRFState(loop, scheduler, lrf, **self._lrf_state_kwargs)
        logger.info("[LearningRateFinder] start: %s, end: %s, steps: %s",
                    self._scheduler_kwargs["start_lr"],
                    self._scheduler_kwargs["end_lr"],
                    self._scheduler_kwargs["total_steps"])
        lrf_state.on_load()
        return scheduler, lrf_state

    def on_load(self, loop: TrainStep) -> None:
        """ Initialize LRF by either resuming from checkpoint or starting a fresh sweep

        Detects if should resume an LRF sweep based on session_id==1 AND file_exists==True
        (checkpoint exists) AND iterations<0 (negative indicates still pre-training).

        If this is the main train then the learning rate it set from the stored optimal LR.

        Otherwise runs new LRF sweep: checks if not session 1 or no checkpoint file exist (valid
        for fresh start) to get the optimal learning rate

        We add a reference to Units so that we can remove Self from steppers when we commence full
        training, so that this unit is not accessed again

        Parameters
        ----------
        loop
            Training step providing access to model state, optimizer, events for initialization
        """
        self._units = loop.units
        is_resume = (loop.model.state.session_id == 1 and
                     loop.model.io.file_exists and
                     loop.model.state.iterations < 0)
        if is_resume:
            logger.debug("%s Resuming LRF", self.log_name)

        if not is_resume and (loop.model.state.session_id != 1 or loop.model.io.file_exists):
            self._set_learning_rate_from_lrf(loop.model.state, loop.optimizer_unit)
            return

        self._scheduler, self._lrf_state = self._setup_lr_finder(loop)

    def on_start(self) -> None:
        """ When commencing main training LRF has completed all tasks, so remove this object from
        steppers """
        logger.debug("%s removing self from steppers", self.log_name)
        self._units.stages_optional["step"].remove(self)

    def step(self, iteration: int) -> None:
        """ Perform LRF sweep step if still in initial phase (iteration <= 0).

        For iteration of -1, advances scheduler and checks stopping conditions. If step() returns
        True indicating completion then the unit is cleaned up as we are entering training mode.

        Errors if iteration does not == -1 as this unit removes itself from steppers when entering
        training mode

        Parameters
        ----------
        iteration
            Current training iteration number - negative values indicate pre-training phase
        """
        assert iteration == -1 and self._lrf_state is not None
        if self._lrf_state.step():
            del self._scheduler
            self._scheduler = None

    def state_dict(self) -> dict[str, T.Any]:
        """ Return scheduler's state dictionary for checkpointing

        If we are no longer in pre-train (completed LRF sweep), returns empty dictionary since
        nothing to save. Otherwise delegates to scheduler.state_dict() which includes all LRF-
        specific attributes: beta, total_steps, smooth_losses list,  learning_rates list,
        average_loss tensor, best_loss tensor for resuming an LRF sweep

        Returns
        -------
        Dictionary containing scheduler's full state for restoration later
        """
        if self._scheduler is None:
            return {}
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """ Restore LRF state when resuming from checkpoint

        If scheduler exists and _lrf_state also initialized, logs info level "Resuming" message
        then delegates to scheduler.load_state_dict() which restores all LRF attributes. This
        handles resumption of interrupted sweeps where checkpoint was saved mid-sweep.

        Parameters
        ----------
        state_dict
            Dictionary from torch.load() containing saved LRF state
        """
        if self._scheduler is None:  # Only when in LRF sweep
            logger.debug("%s No LRF scheduler. Not loading state_dict", self.log_name)
            return
        assert self._lrf_state is not None
        logger.info("[LearningRateFinder] Resuming")
        self._scheduler.load_state_dict(state_dict)
        self._lrf_state.resume()


__all__ = get_module_objects(__name__)

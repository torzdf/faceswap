#! /usr/bin/env python3
""" Defines the standard interface that all training units must implement in training

This module provides the foundational contract for creating custom training units within the
Faceswap system. Each unit is a modular component that plugs into the TrainingLoop at specific
lifecycle points (start, save, update, end) and can perform specialized tasks without modifying
core logic
"""
from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

from lib.utils import get_module_objects

if TYPE_CHECKING:
    from lib.training.training_loop import TrainStep


class TrainingUnit(abc.ABC):
    """ Base abstract class defining the contract for all training units.

    All custom units must inherit from this class to participate in the training loop lifecycle.
    The base implementation provides empty methods and properties that child classes can override
    selectively - only implement what your unit needs, leave others untouched.

    Lifecycle method conventions:

    `on_start`(loop)          : Initialize resources when training begins (called once at start)
    `step`(iteration)         : Perform per-iteration work during training loop (every batch)
    `on_save`(iteration)      : Save derived data when checkpoint is saved (periodic save points)
    `on_update`()             : Generate previews or metrics
    `on_end`()                : Cleanup operations when training completes successfully
    `load_state_dict`(state)  : Restore unit state from checkpoint dictionary if needed
    `state_dict`()            : Return unit's persistent state for checkpointing if needed

    Capability detection properties (used by TrainingLoop to route calls):

    `has_start`      -> bool  : ``True`` if `on_start` is overridden in child class
    `has_step`       -> bool  : ``True`` if `step` method is implemented and should be called each iteration
    `has_save`       -> bool  : ``True`` if `on_save` handles checkpoint save events
    `has_update`     -> bool  : ``True`` if periodic batch-level work occurs between iterations
    `has_end`        -> bool  : ``True`` if cleanup operations occur at training completion
    `has_state_dict` -> bool  : ``True`` only if both persistence methods are overridden

    The `log_name` attribute is automatically set during __init__() to "[ClassName]" format for
    consistent logging across all units. Use logger.debug("%s ...", self.log_name) in your code.
    """  # pylint:disable=line-too-long  # noqa[E501]

    def __init__(self) -> None:
        self.log_name = f"[{self.__class__.__name__}]"
        """ Standardized prefix for debug logging """

    def on_start(self, loop: TrainStep) -> None:  # pylint:disable=unused-argument
        """ Initialize resources when training begins

        Override this method to perform one-time setup operations needed at the start of training:
        load configurations, initialize external connections, register with other systems, etc.
        Called exactly once before the first iteration begins.

        Parameters
        ----------
        loop
            The built training step object (available for accessing model state and other units if
            needed)

        Notes
        -----
        Only override this method if your unit requires initialization work. If no setup is needed,
        leave the default implementation which does nothing and will be removed from the training
        loop.
        """
        return

    def step(self, iteration: int) -> None:  # pylint:disable=unused-argument
        """ Perform work during each training iteration

        Override this method to execute per-iteration tasks such as logging metrics, updating
        live previews, writing TensorBoard scalars, or checking early-stopping conditions. Called
        once for every batch processed throughout the entire training session

        Parameters
        ----------
        iteration
            Current iteration number (negative during pre-training phase like LRF)

        Notes
        -----
        The iteration parameter can be negative during special phases (learning rate finding,
        model loading). Check `if iteration > 0:` before performing operations that depend on
        the model being in training state.

        Use cases:
        - Log progress every N iterations: ``if iteration % 10 == 0: logger.info(...)``
        - Update live preview in GUI cache file
        - Write loss curves to TensorBoard
        - Check custom convergence criteria for early stopping
        """
        return

    def on_save(self, iteration: int) -> None:  # pylint:disable=unused-argument
        """ Save derived data when model checkpoint is saved

        Override this to save files that depend on current model state: generate timelapse images,
        write analysis results to CSV, or create any derivative output at save points. Called after
        the main model checkpoint has been prepared but before it's written to disk

        Parameters
        ----------
        iteration
            Current training iteration when save occurs (useful for naming derived files)

        Notes
        -----
        This is ideal for units that need to create output based on trained model weights at
        specific checkpoints. The method runs before the checkpoint file is finalized, so don't
        modify the model itself here - just generate and save derived data.
        """
        return

    def on_update(self) -> None:
        """ Generate previews or metrics at ad-hoc intervals

        Override this to perform periodic tasks that run once per training loop update cycle.
        Updates occur at the first real training batch, at each save iteration and when a user
        requests an update. Typical use case is generating preview images for GUI display
        """
        return

    def on_end(self) -> None:
        """ Perform cleanup operations when training completes.

        Override this to execute finalization tasks: close connections, remove temporary files
        created during training, log completion messages, or release resources. Called whether
        training finishes successfully or is interrupted

        Notes
        -----
        This fires at the very end of training regardless of success/failure. Use it for cleanup
        that should happen in both cases, but be careful not to perform destructive operations
        without checking first.
        """
        return

    def load_state_dict(self,
                        state_dict: dict[str, Any]) -> None:  # pylint:disable=unused-argument
        """ Restore unit's persistent data from checkpoint.

        Override this only if your unit stores custom counters, flags, or parameters that should be
        restored when loading a saved model checkpoint. Most units don't need persistence - the
        TrainingLoop will automatically save everything it manages that has this method.

        Parameters
        ----------
        state_dict
            Dictionary containing persistent data for this unit (check keys carefully)

        Notes
        -----
        This method is only called if BOTH state_dict() and load_state_dict() are overridden in
        your class. The `has_state_dict` property determines whether the TrainingLoop attempts
        persistence operations.
        """
        return

    def state_dict(self) -> dict[str, Any] | None:
        """ Return unit's persistent state as dictionary for checkpointing.

        Override this only if your unit maintains custom data that should be saved alongside model
        checkpoints. Most units don't require persistence - Do not override this if no special save
        is needed. The TrainingLoop saves everything it manages automatically

        Returns
        -------
        State dictionary containing persistent values, or None if not implemented

        Notes
        -----
        This method is only called when BOTH state_dict() and load_state_dict() are overridden.
        """
        return None

    def _is_overriden(self, method_name: str) -> bool:
        """ Check if a specific method has been overridden by the current class.

        Parameters
        ----------
        method_name
            Name of method to check (e.g., "on_start", "step", etc.)

        Returns
        -------
        ``True`` if child class defines its own version, ``False`` otherwise
        """
        child_func = getattr(type(self), method_name)
        base_func = getattr(TrainingUnit, method_name)
        return child_func is not base_func

    @property
    def has_start(self) -> bool:
        """ ``True`` if this unit implements an `on_start` method """
        return self._is_overriden("on_start")

    @property
    def has_step(self) -> bool:
        """ ``True`` if this unit implements a `step` method """
        return self._is_overriden("step")

    @property
    def has_save(self) -> bool:
        """ ``True`` if this unit implements an `on_save` method """
        return self._is_overriden("on_save")

    @property
    def has_update(self) -> bool:
        """ ``True`` if this unit implements an `on_update` method """
        return self._is_overriden("on_update")

    @property
    def has_end(self) -> bool:
        """ ``True`` if this unit implements an `on_end` method """
        return self._is_overriden("on_end")

    @property
    def has_state_dict(self) -> bool:
        """ ``True`` if this unit implements both `state_dict` and `load_state_dict methods """
        return self._is_overriden("load_state_dict") and self._is_overriden("state_dict")


__all__ = get_module_objects(__name__)

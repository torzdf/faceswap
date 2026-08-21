#! /usr/bin/env python3
""" Units are responsible for carrying out an operation at each training step or save interval

This file contains the base TrainerUnit that all units must inherit from
"""
from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING

from lib.utils import get_module_objects

if TYPE_CHECKING:
    from lib.training.training_loop import TrainStep


class TrainingUnit(abc.ABC):
    """ Abstract base class for training loop units

    Units are modular components that execute specific operations during a Faceswap training
    session. They follow a lifecycle of on_start → step → on_save (periodic) → on_end, allowing
    each unit to participate in the training flow independently. Override any combination of these
    methods to customize behavior without affecting other units.

    Notes
    -----
    This class implements a strategy pattern where each derived unit handles its own specific
    responsibility (e.g., loss tracking, model saving, plugin forward/backward passes). The
    TrainStep orchestrates the sequence by calling these lifecycle methods at appropriate times.

    Examples
    --------
    >>> from lib.training.units import LossUnit, SaveUnit
    >>> # LossUnit handles loss computation and NaN protection
    >>> # SaveUnit handles model checkpointing with backup logic
    """
    def __init__(self) -> None:
        self.log_name = f"[{self.__class__.__name__}]"
        """ Standardized prefix for debug logging """

    def on_start(self, loop: TrainStep) -> None:  # pylint:disable=unused-argument
        """ Override to run the unit's initialization when training commences

        This method is called immediately before the model processes its first real batch.
        Use it to establish connections to other units (model, optimizer, loss), initialize
        state variables, or perform setup that depends on the TrainStep context.

        Parameters
        ----------
        loop
            The configured training loop object about to process its first batch. Access
            shared attributes like _device, current_loss, and iteration tracking from here.

        Notes
        -----
        If your unit doesn't require any initialization at this stage, simply inherit without
        overriding.
        """
        return

    def step(self, iteration: int) -> None:  # pylint:disable=unused-argument
        """ Override to run the unit's per-iteration processing after backpropagation

        This method is called once per training batch, following the backward pass and
        optimizer update. It's where units process loss values, track metrics, or perform
        operations that happen every iteration rather than at save intervals.

        Parameters
        ----------
        iteration
            The current total iteration being processed. Use this for logging, timing measurements,
            or conditional logic based on training progress.

        Notes
        -----
        Set iteration = -1 during learning rate warmup phases. Check this value in your
        override to skip operations that shouldn't run until full training has started.
        """
        return

    def on_save(self, iteration: int) -> None:  # pylint:disable=unused-argument
        """ Override to run the unit's actions when a model save occurs

        This method is called at configured save intervals (e.g., every 250 iterations).
        Use it to log metrics, output contributions, perform backups, or trigger external
        monitoring tools. Unlike step(), this runs regardless of iteration count - only
        the interval or manual intervention determines frequency. If your unit doesn't need any
        to perform any actions during a save iteration then don't override this method.

        Parameters
        ----------
        iteration
            The total iteration number when save occurs (e.g., 250, 500, 750...). Use for
            checkpoint naming or version tracking if needed.

        Notes
        -----
        This method is called even during learning rate finder runs. Be careful to not
        perform operations that might interfere with LRF functionality (like resetting state).
        """
        return

    def on_update(self) -> None:
        """ Override to run the unit's actions when training state is updated

        This method is called by either a save iteration or manual intervention. It allows units
        to doesn't run every iteration and unlike on_save(), it's not tied to weight saving.

        Notes
        -----
        Currently only save iterations and manual keypresses trigger the update flag, so this is
        used for updating previews.
        """
        return

    def on_end(self) -> None:
        """ Override to run the unit's cleanup actions when training completes

        Called after all batches are processed and final saves occur. Use it for resource
        cleanup, logging final statistics, or closing external connections opened during
        training. If your unit doesn't need post-training cleanup, don't override this method.

        Notes
        -----
        This is the last opportunity to log information or perform cleanup before the
        TrainStep exits. Consider logging final metrics here for reporting purposes.
        """
        return

    def load_state_dict(self,
                        state_dict: dict[str, Any]) -> None:  # pylint:disable=unused-argument
        """ Restore unit's internal state from a serialized dictionary representation.

        Override to restore saved model checkpoints or unit-specific state (e.g., iteration
        counters, buffers, cached values) after loading. By default, this method performs no
        operation but can be implemented in derived units that require state persistence across
        training sessions.

        Parameters
        ----------
        state_dict
            A dictionary containing serialized unit state data with keys matching the attributes
            restored by this class's ``state_dict`` method. Keys should correspond to instance
            variables or nested sub-state objects (e.g., model weights, optimizer state).

        Notes
        -----
        This method is only active if both this and ``state_dict`` are overridden in a derived unit
        class.

        Examples
        --------
        >>> # Derived unit with state persistence:
        >>> class CustomUnit(TrainingUnit):
        ...     def __init__(self, counter: int = 0):
        ...         self.counter = counter
        ...
        ...     def state_dict(self) -> dict[str, Any]:
        ...         return {"counter": self.counter}
        ...
        ...     def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...         self.counter = state_dict["counter"]
        """
        return

    def state_dict(self) -> dict[str, Any] | None:
        """ Serialize unit's internal state to a dictionary representation for checkpointing.

        Override to persist training progress by capturing model weights, optimizer states, or
        unit-specific variables into a serializable format. The default implementation returns
        ``None`` but derived units should implement custom serialization logic for stateful
        components (e.g., cached buffers, counters, intermediate computations).

        Returns
        -------
        A dictionary containing serialized state data with keys matching what will be loaded
        by this class's ``load_state_dict`` method. Returns ``None`` if the unit doesn't
        implement custom state serialization in a derived class. Default: ``None``

        Notes
        -----
        This method is only meaningful when overridden in a derived unit class. Check the
        ``has_state_dict`` property before attempting to save state to ensure the derived
        unit implements custom state serialization logic. The returned dictionary should be
        compatible with PyTorch's checkpoint format for model/optimizer weights, plus any
        additional keys needed for unit-specific state restoration.

        Examples
        --------
        >>> # Derived unit with state persistence:
        >>> class CustomUnit(TrainingUnit):
        ...     def __init__(self, counter: int = 0):
        ...         self.counter = counter
        ...
        ...     def state_dict(self) -> dict[str, Any]:
        ...         return {"counter": self.counter}
        ...
        ...     def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...         self.counter = state_dict["counter"]
        """
        return None

    def _is_overriden(self, method_name: str) -> bool:
        """ Check whether a child class has overridden a specific lifecycle method

        Parameters
        ----------
        method_name
            The method name to check (e.g., "on_start", "step", "on_save", "on_end")

        Returns
        -------
        ``True`` if the derived unit implements its own version of this method, ``False`` otherwise

        Notes
        -----
        This is used internally by the TrainStep to determine which units need lifecycle
        hooks. It checks if getattr(type(self), method_name) differs from the base class's
        implementation using function identity comparison.
        """
        child_func = getattr(type(self), method_name)
        base_func = getattr(TrainingUnit, method_name)
        return child_func is not base_func

    @property
    def has_start(self) -> bool:
        """ ``True`` if the unit implements an `on_start` method """
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
        """ ``True`` if this unit implements a state_dict """
        return self._is_overriden("load_state_dict") and self._is_overriden("state_dict")


__all__ = get_module_objects(__name__)

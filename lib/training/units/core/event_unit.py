#!/usr/bin/env python3
""" Handles control flow of internal and external event triggers. """
from __future__ import annotations

import logging
import typing as T

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    from lib.training.training_loop import TrainingEvents

logger = logging.getLogger(__name__)


class EventUnit(TrainingUnit):
    """ Handles control flow of internal and external event triggers.

    This unit monitors and processes event signals passed through the ``TrainingEvents`` object,
    enabling communication between the training thread and GUI/main process. It specifically
    handles mask toggle events by automatically triggering an update when a toggle is requested
    but no pending updates exist.

    The unit checks two conditions at each iteration: if a mask toggle has been queued and
    no update is currently pending, it sets the update flag to allow processing of the toggle.

    Parameters
    ----------
    events
        The TrainingEvents instance containing thread-safe event flags for communication
        between the training loop and external triggers (GUI, callbacks). Must be a valid
        TrainingEvents object with all required Event fields initialized.
    save_interval
        Used for when in pre-train mode to trigger the preview update every save interval

    Notes
    -----
    This unit is designed to be used as a core unit in the training pipeline. It should
    always be registered via TrainStep._create_base_units() or equivalent factory methods.

    Examples
    --------
    >>> # EventUnit is automatically created in TrainStep._create_base_units()
    >>> event_unit = EventUnit(events=training_events)
    >>> units.add_unit(event_unit, is_core=True)
    """
    def __init__(self, events: TrainingEvents, save_interval: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._events = events
        self._save_interval = save_interval
        self._pre_train_steps = 0

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"events={self._events!r}, "
                f"save_interval={self._save_interval!r})")

    def _handle_pretrain(self) -> None:
        self._pre_train_steps += 1
        if self._pre_train_steps % self._save_interval == 0:
            logger.debug("%s Pre-train save iteration %s. Calling events.update.set()",
                         self.log_name, self._pre_train_steps)
        self._events.update.set()

    def step(self, iteration: int) -> None:
        """ Execute per-iteration event processing.

        This method is called once per training batch after the backward pass and optimizer
        update completes. It handles mask toggle events by automatically triggering an update
        when a toggle has been requested but no pending updates exist.

        Parameters
        ----------
        iteration
            The current total iteration being processed. Not used in this implementation
            as event handling is independent of training progress.

        Notes
        -----
        The conditional logic checks if a mask toggle was queued but no update is pending:

        - If **both** conditions are true, the update flag is set to process the toggle
        - This prevents multiple updates from being triggered in rapid succession
        - Toggle events typically originate from GUI, Preview or Cli keypress interactions
        """
        if iteration < 0:
            self._handle_pretrain()

        if self._events.toggle_mask.is_set() and not self._events.update.is_set():
            logger.debug("[EventsUnit] Triggering update for mask toggle [%s]", iteration)
            self._events.update.set()


__all__ = get_module_objects(__name__)

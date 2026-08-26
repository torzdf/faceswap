#! /usr/bin/env python3
""" Event handling unit for training loop operations.

This module provides the core EventUnit class which handles various event-based operations during
training including pre-training steps, save interval management, and mask toggle events.
"""
from __future__ import annotations

import logging
import typing as T

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    from lib.training import TrainingEvents

logger = logging.getLogger(__name__)


class EventUnit(TrainingUnit):
    """ Handle event-driven operations during training iterations.

    Monitors and processes event signals passed through the
    :class:`lib.training.training_loop.TrainingEvents` object, enabling communication between the
    training thread and GUI/main process

    Parameters
    ----------
    events
        The shared events object for coordinating between different parts of the training loop, and
        between the training thread and the main thread
    """
    def __init__(self, events: TrainingEvents) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._events = events

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}(events={self._events!r})"

    def step(self, iteration: int) -> None:
        """ Execute one training step for event handling operations.

        This method is called once per training batch after the backward pass and optimizer
        update completes. The following events are processed:
           - Triggers update events when mask toggle is requested and no other updates are pending

        Parameters
        ----------
        iteration
            Current training iteration number. Negative values indicate pre-training phase
        """
        if self._events.toggle_mask.is_set() and not self._events.update.is_set():
            logger.debug("[EventsUnit] Triggering update for mask toggle [%s]", iteration)
            self._events.update.set()


__all__ = get_module_objects(__name__)

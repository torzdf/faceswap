#! /usr/bin/env python3
""" Training unit for managing model state during training.

This module contains the core StateUnit class which is responsible for creating a new session and
tracking  iteration counts and session information during the training process. It
integrates with the core training loop to maintain accurate progress tracking and state management.
"""
from __future__ import annotations

import logging
import typing as T

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    from lib.model.plugin import State

logger = logging.getLogger(__name__)


class StateUnit(TrainingUnit):
    """ Manages session creation and training state iteration tracking for the current session.

    This unit is responsible for maintaining and incrementing iteration counters within the
    training state, providing access to both session-specific and total iterations during model
    training.

    Parameters
    ----------
    state
        The training state object that manages iteration counts and session data
    batch_size
        The number of samples processed in each training iteration
    """
    def __init__(self, state: State, batch_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._state = state
        self._batch_size = batch_size
        state.add_new_session(batch_size)

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"state={self._state!r}, "
                f"batch_size={self._batch_size!r})")

    @property
    def session_iteration(self) -> int:
        """ The current session iteration for the currently training session """
        return self._state.session_iterations

    @property
    def iteration(self) -> int:
        """ The current total iteration across all training sessions """
        return self._state.iterations

    def step(self, iteration: int) -> None:  # pylint:disable=unused-argument
        """ Increment the iteration counters for the current session.

        Called at the start of each training iteration to increment both overall and session-
        specific counters to the next step to be processed

        Parameters
        ----------
        iteration
            The current iteration number in the training process
        """
        self._state.increment_iterations()


__all__ = get_module_objects(__name__)

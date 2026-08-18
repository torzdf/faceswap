#!/usr/bin/env python3
""" Handles updating of training total and session iteration counts. """
from __future__ import annotations

import logging
import typing as T

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    from lib.model.plugin.train_state import State

logger = logging.getLogger(__name__)


class StateUnit(TrainingUnit):
    """ Handles creation of a new session, and updating of training total and session iteration
    counts

    This unit manages tracking both overall training iterations and per-session progress.

    Parameters
    ----------
    state
        The State object containing iteration tracking data
    batch_size
        Number of samples processed per iteration in this workflow

    Attributes
    ----------
    session_iteration
        Current iteration count for the active training session
    iteration
        Total cumulative iterations across all sessions
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
        """ Current iteration count for the active training session """
        return self._state.session_iterations

    @property
    def iteration(self) -> int:
        """ The total number of iterations processed since training began """
        return self._state.iterations

    def step(self, iteration: int) -> None:
        """ Advance the iteration counter by one

        This method should be called at the start of each training loop to increment
        both overall and session-specific counters to the next step to be processed

        Parameters
        ----------
        iteration
            The current iteration value (stored for reference/logging purposes)
        """
        self._state.increment_iterations()


__all__ = get_module_objects(__name__)

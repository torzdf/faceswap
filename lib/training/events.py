#!/usr/bin/env python3
"""TrainingEvents - Handles event communication within the training thread and between the main
thread and the training thread """
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from threading import Event, Lock
import typing as T

import numpy as np

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class TrainingEvents:
    """ Event system enabling communication between training thread and main process

    This dataclass provides a synchronized event mechanism for coordinating operations across
    threads. The training loop runs in a background FSThread, while the monitor runs on the main
    thread. Events allow the monitor to request checkpoints (save), preview updates (update),
    mask toggles, or exit training without blocking the training loop

    Thread Safety:
    -----------
    All event operations are protected by a threading.Lock preventing race conditions when
    accessing preview (shared between threads). The lock ensures atomic reads/writes of the
    preview data which can be large numpy arrays

    Preview Support:
    ------------
    The training preview is held internally in this object. `get_preview()` retrieves and clears
    this atomically, while `set_preview()` stores previews for the next iteration to retrieve from
    the main loop
    """
    save: Event = field(default_factory=Event)
    """ Event object signaling checkpoint saving request from the main thread """
    exit: Event = field(default_factory=Event)
    """ Event object requesting immediate training termination (after save) """
    update: Event = field(default_factory=Event)
    """ Event object requesting preview refresh or status update """
    toggle_mask: Event = field(default_factory=Event)
    """ Event object for mask inversion operations during training visualization """
    _preview: None | npt.NDArray[np.uint8] = None
    _lock: Lock = field(default_factory=Lock)

    def get_preview(self) -> None | npt.NDArray[np.uint8]:
        """ Retrieve and clear the latest preview image generated during training

        Atomically reads and clears the preview buffer to prevent multiple calls from blocking each
        other. Subsequent calls will return None until set_preview() is called with a new preview
        in the training thread

        Returns
        -------
        The latest preview image (BGR, uint8, (H, W, C)) or ``None`` if no preview available
        """
        with self._lock:
            if self._preview is None:
                return None
            retval = self._preview
            self._preview = None
        logger.debug("[TrainingEvents] Getting preview: %s", retval.shape)
        return retval

    def set_preview(self, preview: npt.NDArray[np.uint8]) -> None:
        """ Store a preview image for retrieval by next `get_preview()` call

        Called by the training thread to render a preview image for display by the main thread

        Parameters
        ----------
        preview
            The latest preview image (BGR, uint8, (H, W, C))
        """
        logger.debug("[TrainingEvents] Setting preview: %s", preview.shape)
        with self._lock:
            self._preview = preview


__all__ = get_module_objects(__name__)

#!/usr/bin/env python3
""" Preview plumbing that passes composed frames from the training thread to display backends, plus
an abstract PreviewBase subclassed by GUI and OpenCV renderers """
from __future__ import annotations
import logging
import typing as T

from threading import Event, Lock
from time import sleep

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from collections.abc import Generator
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)

TriggerType = dict[T.Literal["toggle_mask", "refresh", "save", "quit", "shutdown"], Event]
""" Mapping of trigger name to the threading event a display backend uses to request an action """

TriggerKeysType = T.Literal["m", "r", "s", "enter"]
""" Physical keys mapped by the base viewer onto preview actions """

TriggerNamesType = T.Literal["toggle_mask", "refresh", "save", "quit"]
""" Human-readable names for the four supported preview actions; mapping to `TriggerType` event """


class PreviewBuffer():
    """ Thread-safe store of named preview images consumed by a display backend

    Previews live in a ``{name: image}`` mapping guarded by a lock, so the training thread that
    produces them can hand off without blocking the viewer. A single update flag is set when an
    image lands and cleared once drained, so waiters wake only for genuinely new data instead
    of spinning on stale copies
    """
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        self._images: dict[str, npt.NDArray[np.uint8]] = {}
        self._lock = Lock()
        self._updated = Event()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}()"

    @property
    def is_updated(self) -> bool:
        """ ``True`` if an image has been added since the last drain; ``False`` otherwise """
        return self._updated.is_set()

    def add_image(self, name: str, image: npt.NDArray[np.uint8]) -> None:
        """ Record one preview under a name and flag it available for display backends

        Store the image in the backing mapping (guarded by the lock) then set the update flag so
        any waiter blocked on `get_images` wakes immediately. This is the only way to make newly
        produced previews visible without restarting the viewer

        Parameters
        ----------
        name
            Identifier used later to retrieve and render this preview
        image
            The composed sample frame to store, an RGB ``numpy`` array of one fixed shape across
            the run
        """
        logger.debug("[PreviewBuffer] Adding image: (name: '%s', shape: %s)", name, image.shape)
        with self._lock:
            self._images[name] = image
        logger.debug("[PreviewBuffer] Added images: %s", list(self._images))
        self._updated.set()

    def get_images(self) -> Generator[tuple[str, npt.NDArray[np.uint8]], None, None]:
        """ Yield every currently-stored preview once and clear its update flag

        Walk the backing mapping in place while holding the lock for the whole pass so callers
        receive ``(name, image)`` pairs together, then reset the update flag so the next
        `is_updated` check stays false until new data lands.

        Yields
        ------
        name
            The name of the preview image
        image
            The preview image
        """
        logger.debug("[PreviewBuffer] Retrieving images: %s", list(self._images))
        with self._lock:
            for name, image in self._images.items():
                logger.debug("[PreviewBuffer] Yielding: '%s' (%s)", name, image.shape)
                yield name, image
            if self.is_updated:
                logger.debug("[PreviewBuffer] Clearing updated event")
                self._updated.clear()
                logger.debug("[PreviewBuffer] Retrieved images")


class PreviewBase():  # pylint:disable=too-few-public-methods
    """ Abstract base class for rendering training previews into a display window

    Concrete backends — such as the Tkinter GUI and the OpenCV fallback — implement
    `_display_preview`; this base supplies the shared keymap, shutdown check and launch loop they
    all rely on. It never builds images itself; it only pulls already-composed previews from
    `PreviewBuffer` and decides whether a fresh render is due before delegating to the subclass
    renderer

    Parameters
    ----------
    preview_buffer
        The thread-safe store of named previews being rendered; new frames are read through its
        `get_images`
    triggers, optional
        Optional mapping of trigger name to threading event used to signal this viewer (for example
        refresh or save). When ``None``, only the private shutdown hook is honored and key-driven
        triggers are unavailable. Default: ``None``
    """
    def __init__(self,
                 preview_buffer: PreviewBuffer,
                 triggers: TriggerType | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._triggers = triggers
        self._buffer = preview_buffer
        self._keymaps: dict[TriggerKeysType, TriggerNamesType] = {"m": "toggle_mask",
                                                                  "r": "refresh",
                                                                  "s": "save",
                                                                  "enter": "quit"}
        self._title = ""

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"preview_buffer={self._buffer!r}, "
                f"triggers={self._triggers!r})")

    @property
    def _should_shutdown(self) -> bool:
        """ ``True`` when a shutdown signal is active; ``False`` while to keep running """
        if self._triggers is None or not self._triggers["shutdown"].is_set():
            return False
        logger.debug("[%s] Shutdown signal received", self.__class__.__name__)
        return True

    def _launch(self) -> None:
        """ Wait for one new preview image, then hand rendering to the subclass backend """
        logger.debug("[%s] Launching", self.__class__.__name__)
        while True:
            if self._should_shutdown:
                logger.debug("[%s] Shutdown received", self.__class__.__name__)
                return
            if not self._buffer.is_updated:
                logger.debug("[%s] Waiting for preview image", self.__class__.__name__)
                sleep(1)
                continue
            break
        logger.debug("[%s] Launching preview", self.__class__.__name__)
        self._display_preview()

    def _display_preview(self) -> None:
        """ Override to render a single preview frame. Subclasses such as `PreviewTk` and
        `PreviewCV` implement how the next image actually reaches the screen. """
        raise NotImplementedError()


__all__ = get_module_objects(__name__)

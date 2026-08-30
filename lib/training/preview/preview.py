#!/usr/bin/env python3
from __future__ import annotations
import logging
import typing as T

from threading import Event, Lock
from time import sleep

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from collections.abc import Generator
    import numpy as np

logger = logging.getLogger(__name__)

TriggerType = dict[T.Literal["toggle_mask", "refresh", "save", "quit", "shutdown"], Event]
TriggerKeysType = T.Literal["m", "r", "s", "enter"]
TriggerNamesType = T.Literal["toggle_mask", "refresh", "save", "quit"]

class PreviewBuffer():
    def __init__(self) -> None:
        logger.debug("Initializing: %s", self.__class__.__name__)
        self._images: dict[str, np.ndarray] = {}
        self._lock = Lock()
        self._updated = Event()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def is_updated(self) -> bool:
        return self._updated.is_set()

    def add_image(self, name: str, image: np.ndarray) -> None:
        logger.debug("Adding image: (name: '%s', shape: %s)", name, image.shape)
        with self._lock:
            self._images[name] = image
        logger.debug("Added images: %s", list(self._images))
        self._updated.set()

    def get_images(self) -> Generator[tuple[str, np.ndarray], None, None]:
        logger.debug("Retrieving images: %s", list(self._images))
        with self._lock:
            for name, image in self._images.items():
                logger.debug("Yielding: '%s' (%s)", name, image.shape)
                yield name, image
            if self.is_updated:
                logger.debug("Clearing updated event")
                self._updated.clear()
                logger.debug("Retrieved images")


class PreviewBase():  # pylint:disable=too-few-public-methods
    def __init__(self,
                 preview_buffer: PreviewBuffer,
                 triggers: TriggerType | None = None) -> None:
        logger.debug("Initializing %s parent (triggers: %s)",
                     self.__class__.__name__, triggers)
        self._triggers = triggers
        self._buffer = preview_buffer
        self._keymaps: dict[TriggerKeysType, TriggerNamesType] = {"m": "toggle_mask",
                                                                  "r": "refresh",
                                                                  "s": "save",
                                                                  "enter": "quit"}
        self._title = ""
        logger.debug("Initialized %s parent", self.__class__.__name__)

    @property
    def _should_shutdown(self) -> bool:
        if self._triggers is None or not self._triggers["shutdown"].is_set():
            return False
        logger.debug("Shutdown signal received")
        return True

    def _launch(self) -> None:
        logger.debug("Launching %s", self.__class__.__name__)
        while True:
            if self._should_shutdown:
                logger.debug("Shutdown received")
                return
            if not self._buffer.is_updated:
                logger.debug("Waiting for preview image")
                sleep(1)
                continue
            break
        logger.debug("Launching preview")
        self._display_preview()

    def _display_preview(self) -> None:
        raise NotImplementedError()


__all__ = get_module_objects(__name__)


__all__ = get_module_objects(__name__)

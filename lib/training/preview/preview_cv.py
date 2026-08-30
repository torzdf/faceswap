#!/usr/bin/env python3
from __future__ import annotations
import logging
import typing as T

import cv2

from lib.utils import get_module_objects

from .preview import PreviewBase

if T.TYPE_CHECKING:
    from .preview import PreviewBuffer, TriggerNamesType, TriggerType

logger = logging.getLogger(__name__)



class PreviewCV(PreviewBase):  # pylint:disable=too-few-public-methods
    def __init__(self,
                 preview_buffer: PreviewBuffer,
                 triggers: TriggerType) -> None:
        logger.debug("Unable to import Tkinter. Falling back to OpenCV")
        super().__init__(preview_buffer, triggers=triggers)
        self._triggers: TriggerType = self._triggers
        self._windows: list[str] = []

        self._lookup: dict[int, TriggerNamesType] = {ord(key): val
                                                     for key, val in self._keymaps.items()
                                                     if key != "enter"}
        self._lookup[ord("\n")] = self._keymaps["enter"]
        self._lookup[ord("\r")] = self._keymaps["enter"]

        self._launch()

    @property
    def _window_closed(self) -> bool:
        retval = any(cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1 for win in self._windows)
        if retval:
            logger.debug("Window closed detected")
        return retval

    def _check_keypress(self, key: int):
        if not key or key == -1 or key not in self._lookup:
            return

        if key == ord("r"):
            print("\x1b[2K", end="\r")  # clear last line
            logger.info("Refresh preview requested...")

        self._triggers[self._lookup[key]].set()
        logger.debug("Processed keypress '%s'. Set event for '%s'", key, self._lookup[key])

    def _display_preview(self):
        while True:
            if self._buffer.is_updated or self._window_closed:
                for name, image in self._buffer.get_images():
                    logger.debug("showing image: '%s' (%s)", name, image.shape)
                    cv2.imshow(name, image)
                    self._windows.append(name)

            key = cv2.waitKey(1000)
            self._check_keypress(key)

            if self._triggers["shutdown"].is_set():
                logger.debug("Shutdown received")
                break
        logger.debug("%s shutdown", self.__class__.__name__)


__all__ = get_module_objects(__name__)

"""Manages threads inside Faceswap extraction runners"""
from __future__ import annotations
import logging
import typing as T

from lib.multithreading import FSThread
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class PluginThreads:
    """Handles the holding of threads that will run a plugin's various subprocesses.

    Parameters
    ----------
    name
        The name of the plugin that the threads are being created for
    """
    def __init__(self, name: str) -> None:
        self._name = name
        self._threads: dict[str, FSThread] = {}

    @property
    def enabled(self) -> list[str]:
        """The thread names that have been registered within this group"""
        return [x for x in self._threads if x != "_monitor"]

    def __repr__(self) -> str:
        """Pretty print for logging"""
        obj = f"{self.__class__.__name__}(name={self._name})"
        threads = list(self._threads)
        alive = [x.is_alive() for x in self._threads.values()]
        error = [bool(x.err) for x in self._threads.values()]
        info = f"[threads: {threads}, alive: {alive}, error: {error}]"
        return f"{obj} {info}"

    def register_thread(self, name: str, target: T.Callable[[None], None]) -> None:
        """Register a thread

        Parameters
        ----------
        name
            The name of the plugin's process that is running the thread
        target
            The function to run within the thread
        """
        full_name = f"{self._name}.{name}"
        logger.debug("[%s] Registering thread: '%s'", self._name, name)
        self._threads[name] = FSThread(target=target, name=full_name)

    def start(self) -> None:
        """Start the plugin's threads"""
        for key, thread in self._threads.items():
            logger.debug("[%s] Starting thread: '%s'", self._name, key)
            thread.start()

    def join(self) -> None:
        """Join all of the plugin's threads"""
        for key, thread in self._threads.items():
            logger.debug("[%s] Joining thread: '%s'", self._name, key)
            thread.join()


__all__ = get_module_objects(__name__)

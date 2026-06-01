#!/usr/bin/env python3
"""Saving and loading routines for Faceswap models"""
from __future__ import annotations

import logging
import os
import typing as T

import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from lib.model.state import FaceswapModel

logger = logging.getLogger(__name__)


class ModelIO:
    """Handles loading and saving of a Faceswap model's state_dicts

    Parameters
    ----------
    model
        The Faceswap Model object containing the corresponding state_dicts
    model_dir
        The full path to the model save folder
    """
    def __init__(self, model: FaceswapModel, model_dir: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = f"[{self.__class__.__name__}.{model.name}]"

        self._model = model
        self._model_dir = model_dir
        self._checkpoint_path = os.path.join(model_dir, f"{model.name}.ckpt")
        self._weights_path = os.path.join(model_dir, f"{model.name}.pth")
        self._load_state_dict()

    def __repr__(self) -> str:
        """Cleaner Logging"""
        return f"{self.__class__.__name__}(model={self._model}, model_dir={repr(self._model_dir)})"

    @property
    def _file_exists(self) -> bool:
        """``True`` if a save file exists otherwise ``False``"""
        return any(os.path.isfile(x) for x in (self._checkpoint_path, self._weights_path))

    def _get_latest_save(self) -> str | None:
        """Obtain the latest model's .ckpt or .pth file

        Returns
        -------
        The full path to the latest checkpoint or weights file. ``None`` if no file found
        """
        if not self._file_exists:
            return None

        file_list = (self._checkpoint_path, self._weights_path)
        m_times = [os.path.getmtime(x) if os.path.isfile(x) else 0 for x in file_list]
        retval = file_list[m_times.index(max(m_times))]
        logger.debug("%s Latest save from %s: %s", self._name, file_list, retval)
        return retval

    def _load_state_dict(self) -> None:
        """Load the model checkpoint data from disk to CPU into the FaceswapModel"""
        filename = self._get_latest_save()
        if filename is None:
            logger.debug("%s No save files exist. Not loading", self._name)
            self._model.load_state_dict({})  # Always call it to initialize the plugin
            return

        state_dict: dict[T.Literal["model", "state", "optimizer", "version"],
                         float | dict[str, T.Any]] = torch.load(filename,
                                                                map_location="cpu",
                                                                weights_only=True)
        logger.info("Loaded model from disk: '%s'", filename)
        logger.debug("%s Loaded state_dict version %s. Keys: %s",
                     self._name, state_dict.get("version", 0.0), list(state_dict))
        self._model.load_state_dict(state_dict)

    def save(self, save_optimizer: bool) -> None:
        """Save the state_dicts to disk

        Parameters
        ----------
        ``True`` if the optimizer is to be saved otherwise ``False``
        """
        fname = self._checkpoint_path if save_optimizer else self._weights_path
        logger.debug("%s Saving %s: '%s'",
                     self._name,
                     "checkpoint" if save_optimizer else "weights",
                     fname)
        print("\x1b[2K", end="\r")  # Clear last line
        logger.verbose("Saving Model...")  # type:ignore[attr-defined]

        state_dict = self._model.state_dict()
        if not save_optimizer:
            state_dict = {k: v for k, v in state_dict.items() if k != "optimizer"}

        # TODO Remove/update
        import json
        with open(f"{os.path.splitext(fname)[0]}.json", "w") as o_file:
            json.dump(state_dict["state"], o_file, indent=2)
        torch.save(state_dict, fname)

        msg = "[Saved model]"  # TODO
#        if save_average:
#            msg += f" - Average total loss since last save: {save_average:.5f}"
#        if backed_up:
#            msg += " [Model backed up]"
        logger.info(msg)


__all__ = get_module_objects(__name__)

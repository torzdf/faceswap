#!/usr/bin/env python3
"""Saving and loading routines for Faceswap models"""
from __future__ import annotations

import logging
import os
import typing as T

import torch

from lib.model.faceswap import KerasToTorch
from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from .handler import FaceswapModel

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
    def __init__(self, model_name: str, model_dir: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = f"[{self.__class__.__name__}.{model_name}]"
        self._model_dir = model_dir

        self._checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
        self._weights_path = os.path.join(model_dir, f"{model_name}.pth")
        self._legacy_paths = (os.path.join(model_dir, f"{model_name}.keras"),
                              os.path.join(model_dir, f"{model_name}.h5"))

    def __repr__(self) -> str:
        """Pretty print for logging"""
        return (f"{self.__class__.__name__}("
                f"model_name={repr(self._name.rsplit('.', maxsplit=1)[-1])}, "
                f"model_dir={repr(self._model_dir)})")

    @property
    def _legacy_exists(self) -> bool:
        """``True`` if a legacy save file exists otherwise ``False``"""
        return any(os.path.isfile(x) for x in self._legacy_paths)

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

    def load(self, model: FaceswapModel | None = None
             ) -> dict[T.Literal["model", "state", "optimizer", "version"],
                       float | dict[str, T.Any]]:
        """Load the latest state_dict from disk for the faceswap model

        Parameters
        ----------
        model
            The FaceswapModel object configured from disk. This is only used for migrating legacy
            keras weights to torch weights if a torch model does not exist. Default: ``None``.
            Don't port weights

        Returns
        -------
        The state_dicts for the model, state file and optimizer (if it exists)
        """
        filename = self._get_latest_save()
        if filename is None and not self._legacy_exists:
            logger.debug("%s No save files exist. Not loading", self._name)
            return {}

        if filename is None and model is None:
            raise RuntimeError("Legacy keras model found, but torch structure not provided.")
        if filename is None:
            logger.info("%s Migrating weights from Keras model", self._name)
            assert model is not None
            state_dict = KerasToTorch(model,
                                      next(f for f in self._legacy_paths
                                           if os.path.exists(f))).state_dict()
        else:
            state_dict: dict[T.Literal["model", "state", "optimizer", "version"],
                             float | dict[str, T.Any]] = torch.load(filename,
                                                                    map_location="cpu",
                                                                    weights_only=True)
            logger.debug("Loaded model from disk: '%s'", filename)
        logger.debug("%s Loaded state_dict version %s. Keys: %s",
                     self._name, state_dict.get("version", 0.0), list(state_dict))
        return state_dict

    def save(self, model_state: dict[T.Literal["model", "state", "version", "optimizer"],
                                     float | dict[str, T.Any]]) -> None:
        """Save the state_dicts to disk

        Parameters
        ----------
        model_state
            The FaceswapModel state_dict
            Default: ``None``
        """
        is_checkpoint = bool(model_state.get("optimizer"))
        fname = self._checkpoint_path if is_checkpoint else self._weights_path
        logger.debug("%s Saving %s: '%s'",
                     self._name,
                     "checkpoint" if is_checkpoint else "weights",
                     fname)
        print("\x1b[2K", end="\r")  # Clear last line
        logger.verbose("Saving %s...",  # type:ignore[attr-defined]
                       'checkpoint' if is_checkpoint else 'model')

        # TODO Remove/update
        import json
        with open(f"{os.path.splitext(fname)[0]}.json", "w") as o_file:
            json.dump(model_state["state"], o_file, indent=2)

        torch.save(model_state, fname)

        msg = f"[Saved {'checkpoint' if is_checkpoint else 'model'}]"  # TODO
#        if save_average:
#            msg += f" - Average total loss since last save: {save_average:.5f}"
#        if backed_up:
#            msg += " [Model backed up]"
        logger.info(msg)


__all__ = get_module_objects(__name__)

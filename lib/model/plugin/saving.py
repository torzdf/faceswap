#!/usr/bin/env python3
"""Saving and loading routines for Faceswap models"""
from __future__ import annotations

import logging
import os
import typing as T
from shutil import copyfile, copytree, rmtree

import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from .legacy import KerasToTorch

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
        self._model_name = model_name
        self._model_dir = model_dir

        self._checkpoint_path = os.path.join(model_dir, f"{model_name}.ckpt")
        self._weights_path = os.path.join(model_dir, f"{model_name}.pth")
        self._legacy_paths = (os.path.join(model_dir, f"{model_name}.keras"),
                              os.path.join(model_dir, f"{model_name}.h5"))

    def __repr__(self) -> str:
        """Pretty print for logging"""
        return (f"{self.__class__.__name__}("
                f"model_name={self._model_name}, "
                f"model_dir={repr(self._model_dir)})")

    @property
    def _legacy_exists(self) -> bool:
        """``True`` if a legacy save file exists otherwise ``False``"""
        return any(os.path.isfile(x) for x in self._legacy_paths)

    @property
    def file_exists(self) -> bool:
        """``True`` if a save file exists otherwise ``False``"""
        return any(os.path.isfile(x) for x in (self._checkpoint_path, self._weights_path))

    def _get_latest_save(self) -> str | None:
        """Obtain the latest model's .ckpt or .pth file

        Returns
        -------
        The full path to the latest checkpoint or weights file. ``None`` if no file found
        """
        if not self.file_exists:
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
                                     float | dict[str, T.Any]]) -> bool:
        """Save the state_dicts to disk

        Parameters
        ----------
        model_state
            The FaceswapModel state_dict

        Returns
        -------
        ``True`` if a .ckpt was saved with optimizer, ``False`` if a .pth was saved with just
        weights
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
        return is_checkpoint

    def backup(self) -> None:
        """Backup the latest model save file

        The backed up file is saved with the original filename in the original location with `.bk`
        appended to the end of the name."""
        model_file = self._get_latest_save()
        assert model_file is not None
        backup_file = model_file + ".bk"
        if os.path.exists(backup_file):
            os.remove(backup_file)

        logger.verbose("Backing up: '%s' to '%s'",  # type:ignore[attr-defined]
                       model_file, backup_file)
        copyfile(model_file, backup_file)

    def snapshot(self,
                 iterations: int,
                 state_dict: dict[T.Literal["model", "state", "version", "optimizer"],
                                  float | dict[str, T.Any]]) -> None:
        """Create a full .ckpt snapshot for the given iterations.

        Copies the current tensorboard logs folder to a new snapshot location and saves a full
        ckpt into that folder

        Parameters
        ----------
        iterations
            The number of iterations that this snapshot is being taken at
        state_dict
            The full FaceswapModel checkpoint state_dict
        """
        logger.info("[Snapshot] Creating model snapshot...")
        src = self._model_dir
        dst = f"{src}_snapshot_{iterations}_iters"
        if os.path.isdir(dst):
            logger.debug("[ModelIO] Removing previously existing snapshot folder: '%s'", dst)
            rmtree(dst)
        os.makedirs(dst)

        logs = f"{self._model_name}_logs"
        if os.path.exists(os.path.join(src, logs)):
            logger.debug("[ModelIO] Copying logs for snapshot: '%s'", os.path.join(dst, logs))
            copytree(os.path.join(src, logs), os.path.join(dst, logs))

        fname = os.path.join(dst, os.path.basename(self._checkpoint_path))
        logger.debug("[ModelIO] Saving snapshot: '%s'", fname)
        torch.save(state_dict, fname)

        logger.info("[Snapshot] %s iterations. Saved", iterations)


__all__ = get_module_objects(__name__)

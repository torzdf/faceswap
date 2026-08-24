#! /usr/bin/env python3
""" Training units for managing model saving and loading operations during training

This module contains the core training units responsible for handling model checkpointing, state
persistence, and recovery operations. It includes units for loading saved states and saving
checkpoints at regular intervals
"""
from __future__ import annotations

import logging
import os
from shutil import copyfile, copytree, rmtree
import typing as T

import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from lib.model.plugin import FaceswapModel
    from lib.training.training_loop import TrainingEvents, TrainStep
    from .optimizer_unit import OptimizerUnit

logger = logging.getLogger(__name__)


# TODO need to check why save runs twice when exit is on a save iter

class LoadUnit(TrainingUnit):
    """ Loads previously saved model states and configurations

    This unit is responsible for restoring model state from saved checkpoint files, including
    loading weights, optimizer states, and any additional training metadata that was previously
    stored

    Parameters
    ----------
    model
        The faceswap model that contains the loaded the state_dict information
    """
    def __init__(self, model: FaceswapModel) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._model = model

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}(model={self._model!r})"

    def on_start(self, loop: TrainStep) -> None:
        """ Load saved states from checkpoint files

        Restores optimizer and other training unit information from previously saved checkpoint
        files that were stored in the model's extra state collection

        Parameters
        ----------
        loop
            The training step object managing this unit's lifecycle
        """
        loadable = loop.units.have_state_dict
        logger.debug("%s got loadables: %s", self.log_name, loadable)
        for name, unit in loadable.items():
            state_dict = self._model.pop_extra_state(name)
            if not state_dict:  # TODO handle
                logger.debug("%s Skipping missing state_dict: '%s'", self.log_name, name)
                continue
            logger.debug("%s Loading state_dict: '%s'", self.log_name, name)
            unit.load_state_dict(state_dict)

        self._model.clear_extra_state()


class SaveUnit(TrainingUnit):
    """ Saves model checkpoints and final models during training

    This unit manages the saving of trained models at specified intervals, including creating
    regular checkpoints, backup copies when loss improves, and saving via user input

    Parameters
    ----------
    model
        The FaceswapModel object containing the neural network, state and info for the model
    optimizer
        The optimizer unit containing optimizer state information
    events
        The event system for coordinating training operations
    average_loss
        The array that will hold the total average loss for the current save iteration. The object
        is a 0-dimensional numpy array that gets updated each save iteration so a reference can be
        held
    save_interval
        Number of iterations between regular saves
    snapshot_interval
        Number of iterations between snapshot creation for recovery points
    save_train_state
        When to save training state ("always", "never", or "exit")
    """
    def __init__(self,
                 model: FaceswapModel,
                 optimizer: OptimizerUnit,
                 events: TrainingEvents,
                 average_loss: npt.NDArray[np.float32],
                 save_interval: int,
                 snapshot_interval: int,
                 save_train_state: T.Literal["always", "never", "exit"]) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._events = events
        self._average_loss = average_loss
        self._save_interval = save_interval
        self._snapshot_interval = snapshot_interval
        self._save_state = save_train_state

        self._do_snapshot = False
        self._saveable: dict[str, TrainingUnit]  # set in on_start

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = ", ".join(f"{k[1:]}={v!r}" for k, v in self.__dict__.items()
                           if k in ("_model",
                                    "_optimizer",
                                    "_events",
                                    "_average_loss",
                                    "_save_interval",
                                    "_snapshot_interval",
                                    "_save_state"))
        return f"{self.__class__.__name__}({params})"

    def on_start(self, loop: TrainStep) -> None:
        """ Initialize saveable units and configure saving behavior

        Stores references to all training units that can provide state dictionaries for inclusion
        in saved checkpoints

        Parameters
        ----------
        loop
            The training step object managing this unit's lifecycle
        """
        self._saveable = loop.units.have_state_dict
        logger.debug("%s Stored saveables: %s", self.log_name, self._saveable)

    def step(self, iteration: int) -> None:
        """Check if it's time to trigger a save operation

        Determines whether the current iteration requires saving based on configured save interval
        or snapshot interval and triggers the save event

        Parameters
        ----------
        iteration
            The current iteration number in the training process
        """
        do_save = False
        if iteration > 0 and iteration % self._save_interval == 0:
            logger.debug("%s Save iteration %s", self.log_name, iteration)
            do_save = True
        if iteration > 0 and iteration % self._snapshot_interval == 0:
            logger.debug("%s Snapshot iteration %s", self.log_name, iteration)
            self._do_snapshot = True
            do_save = True
        if do_save:
            logger.debug("%s Calling events.save.set()", self.log_name)
            self._events.save.set()

    def _get_average_loss(self) -> float:
        """ Obtain latest average loss since the last save iteration and set initial average loss

        Returns
        -------
        Average loss value since the last save iteration
        """
        retval = float(self._average_loss)
        if self._model.state.lowest_avg_loss <= 0.0 < retval:
            logger.debug("%s Setting initial lowest average loss: %s", self.log_name, retval)
            self._model.state.lowest_avg_loss = retval
        return retval

    def _backup(self) -> None:
        """ Backup the latest model save file to "{model_name}.bk" """
        model_file = self._model.latest_save
        assert model_file is not None
        backup_file = model_file + ".bk"
        if os.path.exists(backup_file):
            os.remove(backup_file)
        logger.verbose("%s Backing up: '%s' to '%s'",  # type:ignore[attr-defined]
                       self.log_name, model_file, backup_file)
        copyfile(model_file, backup_file)

    def _maybe_backup(self, average_loss: float) -> bool:
        """ Create a backup checkpoint if current loss is better than previous best

        Parameters
        ----------
        average_loss
            The current average loss value

        Returns
        -------
        ``True`` if a backup was created, ``False`` otherwise
        """
        retval = 0.0 < average_loss < self._model.state.lowest_avg_loss
        if not retval:
            return retval
        self._backup()
        logger.debug("%s Updating lowest average loss from: %s, to: %s",
                     self.log_name, self._model.state.lowest_avg_loss, average_loss)
        self._model.state.lowest_avg_loss = average_loss
        return retval

    def _save(self, model_state: dict[str, T.Any]) -> None:
        """ Save the state_dicts to disk

        Parameters
        ----------
        model_state
            The FaceswapModel state_dict
        """
        is_checkpoint = bool(model_state.get("optimizer"))
        fname = self._model.checkpoint_path
        if not is_checkpoint:
            fname = f"{os.path.splitext(fname)[0]}.pth"
        logger.debug("%s Saving %s: '%s'",
                     self.log_name,
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

    def _get_state_dicts(self, is_checkpoint: bool) -> dict[str, T.Any]:
        """ Obtain the state_dicts required for saving

        Parameters
        ----------
        is_checkpoint
            ``True`` to obtain full training state. ``False`` for just model weights

        Returns
        -------
        The model weights, including full training state if `is_checkpoint` is ``True``
        """
        retval = self._model.state_dict()
        if not is_checkpoint:
            return retval
        saveable = {k: v.state_dict() for k, v in self._saveable.items() if v.state_dict()}
        logger.debug("%s Adding saveable units for checkpoint: %s", self.log_name, list(saveable))
        retval |= saveable
        return retval

    def _save_model(self, is_exit: bool) -> bool:
        """ Save the model weights and optionally include optimizer state

        Parameters
        ----------
        is_exit
            Whether this save is happening at training exit

        Returns
        -------
        ``True`` if this was a full checkpoint (not just a model weights)
        """
        is_checkpoint = self._save_state == "always" or (is_exit and self._save_state == "exit")
        state_dict = self._get_state_dicts(is_checkpoint=is_checkpoint)
        self._save(state_dict)
        return is_checkpoint

    def _collate_and_save(self, is_exit: bool) -> None:
        """ Execute the complete save operation for either normal saves or exit

        Parameters
        ----------
        is_exit
            Whether this save is happening at training exit
        """
        average_loss = self._get_average_loss()
        do_backup = self._maybe_backup(average_loss)  # TODO move to after model save?
        is_checkpoint = self._save_model(is_exit)
        msg = f"[Saved {'checkpoint' if is_checkpoint else 'model'}]"
        if average_loss != 0.0:
            msg += f" - Average loss since save: {average_loss:.5f}"
        if do_backup:
            msg += " [Model backed up]"
        logger.info(msg)

    def _snapshot(self) -> None:
        """ Create a full .ckpt snapshot + tensorboard logs for the given iterations """
        logger.info("[Snapshot] Creating model snapshot...")
        src = os.path.dirname(self._model.checkpoint_path)
        iters = self._model.state.iterations
        dst = f"{src}_snapshot_{iters}_iters"
        if os.path.isdir(dst):
            logger.debug("[ModelIO] Removing previously existing snapshot folder: '%s'", dst)
            rmtree(dst)
        os.makedirs(dst)

        logs = f"{self._model.name}_logs"
        if os.path.exists(os.path.join(src, logs)):
            logger.debug("[ModelIO] Copying logs for snapshot: '%s'", os.path.join(dst, logs))
            copytree(os.path.join(src, logs), os.path.join(dst, logs))

        fname = os.path.join(dst, os.path.basename(self._model.checkpoint_path))
        logger.debug("[ModelIO] Saving snapshot: '%s'", fname)
        state_dict = self._get_state_dicts(is_checkpoint=True)
        torch.save(state_dict, fname)

        logger.info("[Snapshot] %s iterations. Saved", iters)

    def on_save(self, iteration: int) -> None:
        """ Execute save operation for the current iteration

        Triggers the full saving process and updates related events for preview
        generation and status reporting

        Parameters
        ----------
        iteration
            The current iteration number in the training process
        """
        if self._events.exit.is_set():
            return  # Handle in clean up
        logger.debug("%s Saving [%s]", self.log_name, iteration)
        self._collate_and_save(False)
        if self._do_snapshot:
            self._snapshot()
            self._do_snapshot = False

        self._events.update.set()  # Trigger preview updates on a save

    def on_end(self) -> None:
        """ Save final model state at training completion

        Performs the final save operation to ensure all training progress is preserved when
        training concludes. This save may include optimizer state depending on configuration
        (``save_optimizer="exit"`` or ``"always"``)
        """
        logger.debug("%s Saving on exit", self.log_name)
        self._collate_and_save(True)


__all__ = get_module_objects(__name__)

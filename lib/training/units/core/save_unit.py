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
from lib.model.plugin.state import StateMarkdown
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

    # ## BACKUP ##
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

    # ## SAVE & SNAPSHOT ##
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

    def _write_model_info(self, model_path: str) -> None:
        """ Write the current model information as markdown file in model folder

        Parameters
        ----------
        model_path
            Full path to the model file that information is being written for
        """
        fname = f"{os.path.splitext(model_path)[0]}_info.md"
        with open(fname, "w", encoding="utf-8", errors="replace") as o_file:
            o_file.write(StateMarkdown(self._model.state).full_summary())

    def _save(self, folder: str, is_checkpoint: bool) -> None:
        """ Save the model weights, optional training state + model info to disk

        Parameters
        ----------
        folder
            The folder to save the model to
        is_checkpoint
            ``True`` if this is a full checkpoint. ``False`` for weights only
        """
        state_dict = self._get_state_dicts(is_checkpoint=is_checkpoint)
        fname = os.path.join(folder, os.path.basename(self._model.checkpoint_path))
        if not is_checkpoint:
            fname = f"{os.path.splitext(fname)[0]}.pth"
        logger.debug("%s Saving %s: '%s'",
                     self.log_name,
                     "checkpoint" if is_checkpoint else "weights",
                     fname)
        logger.verbose("Saving %s...",  # type:ignore[attr-defined]
                       'checkpoint' if is_checkpoint else 'model')

        torch.save(state_dict, fname)
        self._write_model_info(fname)

    # ## SAVE ##
    def _backup_and_save(self, is_exit: bool) -> None:
        """ Execute the complete save operation for either normal saves or exit

        Parameters
        ----------
        is_exit
            Whether this save is happening at training exit
        """
        average_loss = self._get_average_loss()
        do_backup = self._maybe_backup(average_loss)  # TODO move to after model save?
        is_checkpoint = self._save_state == "always" or (is_exit and self._save_state == "exit")

        self._save(os.path.dirname(self._model.checkpoint_path), is_checkpoint)

        msg = f"[Saved {'checkpoint' if is_checkpoint else 'model'}]"
        if average_loss != 0.0:
            msg += f" - Average loss since save: {average_loss:.5f}"
        if do_backup:
            msg += " [Model backed up]"
        logger.info(msg)

    # ## SNAPSHOT ##
    def _get_snapshot_folder(self) -> str:
        """ Create the folder where the next snapshot will be saved

        Returns
        -------
        The full path to the created snapshot folder
        """
        src = os.path.dirname(self._model.checkpoint_path)
        iters = self._model.state.iterations
        retval = f"{src}_snapshot_{iters}_iters"
        if os.path.isdir(retval):
            logger.debug("%s Removing previously existing snapshot folder: '%s'",
                         self.log_name, retval)
            rmtree(retval)
        os.makedirs(retval)
        logger.debug("%s Snapshot folder: '%s'", self.log_name, retval)
        return retval

    def _snapshot_logs(self, destination: str) -> None:
        """ Copy the current log folder to the snapshot folder

        Parameters
        ----------
        destination
            The full path to the destination snapshot folder to copy logs to
        """
        src = os.path.dirname(self._model.checkpoint_path)
        logs = f"{self._model.name}_logs"
        if not os.path.exists(os.path.join(src, logs)):
            return
        logger.debug("%s Copying logs for snapshot: '%s'",
                     self.log_name, os.path.join(destination, logs))
        copytree(os.path.join(src, logs), os.path.join(destination, logs))

    def _snapshot(self) -> None:
        """ Create a full .ckpt snapshot + tensorboard logs for the given iterations """
        logger.info("[Snapshot] Creating model snapshot...")
        dst = self._get_snapshot_folder()
        self._snapshot_logs(dst)
        logger.debug("%s Saving snapshot: '%s'", self.log_name, dst)
        self._save(dst, True)
        logger.info("[Snapshot] Saved: %s iterations", self._model.state.iterations)

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
        self._backup_and_save(False)
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
        self._backup_and_save(True)


__all__ = get_module_objects(__name__)

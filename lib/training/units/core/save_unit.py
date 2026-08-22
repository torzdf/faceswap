#! /usr/bin/env python3
""" Training units for managing model saving and loading operations during training

This module contains the core training units responsible for handling model checkpointing, state
persistence, and recovery operations. It includes units for loading saved states, saving
checkpoints at regular intervals, and creating  snapshots of model progress
"""
from __future__ import annotations

import logging
import typing as T

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from lib.model.plugin.handler import FaceswapModel
    from lib.training.training_loop import TrainingEvents, TrainStep
    from .optimizer_unit import OptimizerUnit

logger = logging.getLogger(__name__)


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
    save_optimizer
        When to save optimizer state ("always", "never", or "exit")
    """
    def __init__(self,
                 model: FaceswapModel,
                 optimizer: OptimizerUnit,
                 events: TrainingEvents,
                 average_loss: npt.NDArray[np.float32],
                 save_interval: int,
                 save_optimizer: T.Literal["always", "never", "exit"]) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._events = events
        self._average_loss = average_loss
        self._save_interval = save_interval
        self._save_optimizer = save_optimizer

        self._saveable: dict[str, TrainingUnit]  # set in on_start

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = ", ".join(f"{k[1:]}={v!r}" for k, v in self.__dict__.items()
                           if k in ("_model",
                                    "_optimizer",
                                    "_events",
                                    "_average_loss",
                                    "_save_interval",
                                    "_save_optimizer"))
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
        and triggers the save event

        Parameters
        ----------
        iteration
            The current iteration number in the training process
        """
        if iteration > 0 and iteration % self._save_interval == 0:
            logger.debug("%s Save iteration %s. Calling events.save.set()",
                         self.log_name, iteration)
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
        self._model.io.backup()
        logger.debug("%s Updating lowest average loss from: %s, to: %s",
                     self.log_name, self._model.state.lowest_avg_loss, average_loss)
        self._model.state.lowest_avg_loss = average_loss
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
        is_checkpoint = self._save_optimizer == "always" or (is_exit and
                                                             self._save_optimizer == "exit")
        saveable = {k: v.state_dict() for k, v in self._saveable.items() if v.state_dict()}
        state_dict = self._model.state_dict() | saveable
        if not is_checkpoint:
            del state_dict["OptimizerUnit"]  # TODO change so optimizer only provides when required
        self._model.io.save(state_dict)
        return is_checkpoint

    def _save(self, is_exit: bool) -> None:
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
        self._save(False)
        self._events.update.set()  # Trigger preview updates on a save

    def on_end(self) -> None:
        """ Save final model state at training completion

        Performs the final save operation to ensure all training progress is preserved when
        training concludes. This save may include optimizer state depending on configuration
        (``save_optimizer="exit"`` or ``"always"``)
        """
        logger.debug("%s Saving on exit", self.log_name)
        self._save(True)


class SnapshotUnit(TrainingUnit):
    """ Creates periodic snapshots of model state for recovery purposes.

    This unit generates snapshot files at regular intervals during training, allowing for recovery
    from specific points in the training process. Snapshots typically contain model weights and
    optimizer states.

    Parameters
    ----------
    model
        The faceswap model to snapshot
    optimizer
        The optimizer unit containing current state information
    snapshot_interval
        Number of iterations between snapshot creation
    """
    def __init__(self, model: FaceswapModel, optimizer: OptimizerUnit, snapshot_interval: int
                 ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._interval = snapshot_interval

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"model={self._model!r}, "
                f"optimizer={self._optimizer!r}, "
                f"snapshot_interval={self._interval!r})")

    def step(self, iteration: int) -> None:
        """ Create a snapshot at specified intervals

        Generates and saves a model snapshot when the current iteration matches the configured
        snapshot interval

        Parameters
        ----------
        iteration
            The current iteration number in the training process
        """
        if iteration < 0 or iteration % self._interval != 0:
            return
        logger.debug("%s Snapshotting model [%s]", self.log_name, iteration)
        state_dict = self._model.state_dict() | {"optimizer": self._optimizer.state_dict()}
        # TODO TB logs will need flushing and state file will be wrong for lowest avg loss +
        # iter count
        self._model.io.snapshot(iteration, state_dict)


__all__ = get_module_objects(__name__)

#!/usr/bin/env python3
""" Handles saving of model weights and checkpoints. """
from __future__ import annotations

import logging
import typing as T

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from . import TrainingUnit

if T.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from lib.model.plugin.handler import FaceswapModel
    from lib.training.training_loop import TrainingEvents
    from .optimizer_unit import OptimizerUnit

logger = logging.getLogger(__name__)


class SaveUnit(TrainingUnit):
    """ Handles saving of model weights and checkpoints

    As well as saving the model, this unit tracks the average loss from each save iteration and
    backs up the model when a new lowest average loss is recorded during training.

    This must be placed last in the ``on_save`` units.

    Parameters
    ----------
    model
        The FaceswapModel object containing the neural network, state and info for the model
    optimizer
        The Optimizer being used to train the model
    average_loss
        The array that will hold the total average loss for the current save iteration. The object
        is a 0-dimensional numpy array that gets updated each save iteration so a reference can be
        held
    save_interval
        The number of iterations between each model save
    save_optimizer
        When to include optimizer state in saved checkpoints. Options are:
        ``"always"``, ``"never"``, or ``"exit"`` (only on training end)
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

    def step(self, iteration: int) -> None:  # pylint:disable=unused-argument
        """ Checks if the current iteration matches the configured save interval and triggers
        saving operations when appropriate. This method is called during each training step
        to determine if the on_save method should be run for units during this iteration

        Parameters
        ----------
        loss
            The current batch loss value (unused - retained for interface compatibility)
        iteration
            The current total iteration count. Saves occur when this is a multiple of
            the configured ``save_interval``.
        """
        if iteration > 0 and iteration % self._save_interval == 0:
            logger.debug("%s Save iteration %s. Calling events.save.set()",
                         self.log_name, iteration)
            self._events.save.set()

    def _get_average_loss(self) -> float:
        """ Obtain the average loss for the current save iteration

        Returns
        -------
        The current average loss for this save iteration as a float. If this is the
        initial loss, it will be stored in ``self._model.state.lowest_avg_loss`` for
        comparison on future saves.
        """
        retval = float(self._average_loss)
        if self._model.state.lowest_avg_loss <= 0.0 < retval:
            logger.debug("%s Setting initial lowest average loss: %s", self.log_name, retval)
            self._model.state.lowest_avg_loss = retval
        return retval

    def _maybe_backup(self, average_loss: float) -> bool:
        """ Perform a model backup if the average loss is lower than any previously seen

        Parameters
        ----------
        average_loss
            The average loss since the last save iteration

        Returns
        -------
        ``True`` if a backup was created (new lowest loss), ``False`` otherwise.
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
            ``True`` if this save occurs when exiting training. If the optimizer is set
            to be saved on exit (via ``save_optimizer="exit"``), it will be included in
            the checkpoint.

        Returns
        -------
        ``True`` if a checkpoint was saved (optimizer state included), ``False`` for
        regular model saves without optimizer.
        """
        is_checkpoint = self._save_optimizer == "always" or (is_exit and
                                                             self._save_optimizer == "exit")
        state_dict = self._model.state_dict()
        if is_checkpoint:
            state_dict["optimizer"] = self._optimizer.state_dict()
        self._model.io.save(state_dict)
        return is_checkpoint

    def _save(self, is_exit: bool) -> None:
        """ Perform the complete save operation for either normal saves or exit

        Parameters
        ----------
        is_exit
            ``True`` if this save occurs when exiting training. This affects whether
            optimizer state is included based on the configuration.
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
        """ Save the model weights

        Parameters
        ----------
        iteration
            The current total iteration count. This is logged for tracking purposes

        Notes
        -----
        Called during normal training flow when saving intervals are reached. Triggers
        loss checking, potential backup, and model save with appropriate logging output.
        """
        if self._events.exit.is_set():
            return  # Handle in clean up
        logger.debug("%s Saving [%s]", self.log_name, iteration)
        self._save(False)
        self._events.update.set()  # Trigger preview updates on a save

    def on_end(self) -> None:
        """ Save the model when training session ends

        Notes
        -----
        Called during cleanup phase of training. This save may include optimizer state
        depending on configuration (``save_optimizer="exit"`` or ``"always"``). Ensures
        progress is preserved even if interrupted at this stage.
        """
        logger.debug("%s Saving on exit", self.log_name)
        self._save(True)


class SnapshotUnit(TrainingUnit):
    """ Creates periodic model Snapshots

    Parameters
    ----------
    model
        The FaceswapModel object containing the neural network, state and info for the model
    optimizer
        The Optimizer being used to train the model
    snapshot_interval
        Number of iterations between snapshots. Set to ``0`` to disable periodic saving,
        or set to a positive integer (e.g., 1000) to save every N iterations.

    Notes
    -----
    This unit performs regular, interval-based saves for long-term preservation and analysis.
    Unlike SaveUnit which backs up on loss improvement, this creates checkpoints at specific
    training milestones regardless of model performance.
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
        """ Process a training step and trigger snapshot if interval is reached

        Parameters
        ----------
        loss
            The current batch loss (unused here - retained for interface compatibility)
        iteration
            The current total iteration count. Snapshots occur when this matches the
            configured ``snapshot_interval``.

        Notes
        -----
        This method is called during each training step. It checks if the iteration
        to save has been reached and, if so, saves a full checkpoint including optimizer state.
        """
        if iteration < 0 or iteration % self._interval != 0:
            return
        logger.debug("%s Snapshotting model [%s]", self.log_name, iteration)
        state_dict = self._model.state_dict() | {"optimizer": self._optimizer.state_dict()}
        # TODO TB logs will need flushing and state file will be wrong for lowest avg loss +
        # iter count
        self._model.io.snapshot(iteration, state_dict)


__all__ = get_module_objects(__name__)

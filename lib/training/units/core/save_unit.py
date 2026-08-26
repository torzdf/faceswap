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
import time
import typing as T

import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .base import TrainingUnit

if T.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from lib.model.plugin import FaceswapModel, State
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

    def on_load(self, loop: TrainStep) -> None:
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
            if not state_dict:
                logger.debug("%s Skipping missing state_dict: '%s'", self.log_name, name)
                continue
            logger.debug("%s Loading state_dict: '%s'", self.log_name, name)
            unit.load_state_dict(state_dict)

        self._model.clear_extra_state()


class StateMarkdown:
    """ Converts model training state into formatted markdown tables for reference

    This utility class transforms the internal State object's data into readable markdown format,
    making it suitable for reading directly or for documentation. It extracts key training
    information including plugin metadata, current session details, loss metrics, learning rate
    findings, and configuration parameters into organized table structures.

    The class provides multiple rendering methods:
        - ``render_model_info()``: Creates a high-level overview table of all important state info
          (model name, version, iterations, sessions count, lowest loss, lr_finder status)
        - ``render_config()``: Formats all configuration parameters from the initial state as
          a key-value table for reference
        - ``render_sessions()``: Generates detailed tables for each training session with batch
          sizes, iteration counts, and timestamps
        - ``full_summary()``: Combines all render methods into a comprehensive markdown report

    Parameters
    ----------
    state
        A State object containing the model's training metadata. This includes plugin version,
        iteration counts, loss values, learning rate finder results, and complete session history.

    Notes
    -----
    The class provides structured output suitable for:
        - Progress reporting during training runs
        - Status checks in terminal interfaces
        - Documentation generation from model checkpoints
        - Comparing configurations across different sessions

    All methods return markdown-formatted strings ready for display or file writing. Session data
    is sorted by ID and only includes completed sessions (those with iterations > 0). Configuration
    parameters are presented as they were in the initial state for reference purposes
    """
    def __init__(self, state: State) -> None:
        logger.debug(parse_class_init(locals()))
        self._state = state

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}(state={self._state!r})"

    @classmethod
    def _format_time(cls, timestamp: float) -> str:
        """ Format a Unix timestamp to human-readable date-time string

        Parameters
        ----------
        timestamp
            Unix epoch time in seconds

        Returns
        -------
        Formatted datetime string in "YYYY-MM-DD HH:MM" format
        """
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))

    @classmethod
    def _format_to_table(cls, data: dict[str, str] | dict[str, list[str]]) -> list[str]:
        """ Format a dictionary of key-value pairs into markdown table rows

        Parameters
        ----------
        data
            Dictionary where keys are column headers (strings) and values are either a strings to
            be displayed in one column or lists of strings as multiple rows for each column

        Returns
        -------
        A list of formatted markdown table rows as strings, ready to be joined with newlines.
        """
        data = {k: v if isinstance(v, list) else [v] for k, v in data.items()}
        col_widths = [max(len(k), *[len(x) for x in v]) for k, v in data.items()]

        lines = [[f" {k.ljust(w)} " if i == 0 else f" {k.rjust(w)} "  # Header
                  for i, (k, w) in enumerate(zip(data, col_widths))]]

        lines.append(["-" * (w + 2) for w in col_widths])  # Break
        lines[-1][1:] = [f"{x[:-1]}:" for x in lines[-1][1:]]  # Justify
        lines.extend([f" {col.ljust(w)} " if i == 0 else f" {col.rjust(w)} "  # Data
                      for i, (col, w) in enumerate(zip(row, col_widths))]
                     for row in zip(*data.values()))

        return ["|" + "|".join(line) + "|" for line in lines]

    def render_model_info(self) -> str:
        """ Generate a summary table of the model's training state

        This method creates a single consolidated markdown table showing key training information
        in an organized format. The table displays model identity, version number, total
        iterations, session count, best loss observed, learning rate finder result, and creation
        timestamp of the first session

        Returns
        -------
        A markdown-formatted string containing a single comprehensive table with model
        information
        """
        state = self._state.state_dict()
        data = {
            "plugin_name": state["plugin_name"],
            "plugin_version": state["plugin_version"],
            "iterations": "Pre-train" if state["iterations"] < 0 else state["iterations"],
            "sessions": len(state["sessions"]),
            "lowest_avg_loss": ("N/A" if not state["lowest_avg_loss"]
                                else f"{state['lowest_avg_loss']:.2e}"),
            "lr_finder": "N/A" if state["lr_finder"] < 0 else f"{state['lr_finder']:.1e}",
            "created": self._format_time(state["sessions"].get(1, {}).get("timestamp",
                                                                          time.time()))
        }
        data = {k.replace("plugin_", "").replace("_", " ").title(): str(v)
                for k, v in data.items()}
        return "\n".join(["### Model Information"] + self._format_to_table(data))

    def render_config(self) -> str:
        """ Format all initial fixed configuration parameters as a markdown table

        Returns a detailed table showing every fixed configuration parameter and its value set on
        model creation

        Returns
        -------
        A markdown-formatted string containing a sorted list of initial configuration parameters
        as a two-column table (Parameter | Value)

        Notes
        -----
        Configuration parameters are shown in their original state from the checkpoint, not
        current values that may have been modified since loading. Useful for reference when
        reviewing saved training runs
        """
        config = self._state.fixed_config
        data = {"Parameter": list(config),
                "Value": [f"{x!r}" for x in config.values()]}
        return "\n".join(["### Model Config"] + self._format_to_table(data))

    def render_sessions(self) -> str:
        """ Generate a history table of all completed training sessions

        This method creates markdown tables showing each session that has been run (including any
        session currently running), with its ID, batch size, iteration count, and creation
        timestamp. Sessions are displayed in reverse chronological order (most recent first).

        Returns
        -------
        A markdown-formatted string containing a series of tables for each training session.
        Each session table has two sections: metadata (ID, batch size, iterations, start time)
        and configuration parameters specific to that run
        """
        state = self._state.state_dict()
        info = [{"Session": str(k),
                 "Batch Size": str(v["batch_size"]),
                 "Iterations": str(v["iterations"]),
                 "Created":  self._format_time(v["timestamp"])}
                for k, v in state["sessions"].items()]
        conf = [{"Parameter": list(v["config"]),
                 "Value": [f"{x!r}" for x in v["config"].values()]}
                for v in state["sessions"].values()]

        lines = []
        for idx, (session, conf) in enumerate(zip(reversed(info), reversed(conf))):
            lines.append(f"### Session {len(state["sessions"]) - idx}")
            lines.append("#### Session Information")
            lines.extend(self._format_to_table(session) + [""])
            lines.append("#### Session Config")
            lines.extend(self._format_to_table(conf) + [""])
        return "\n".join(lines)

    def full_summary(self) -> str:
        """ Generate a complete markdown report combining all state information

        This convenience method combines model information, configuration parameters, and
        training session history into a single comprehensive markdown document. Useful for
        generating complete status reports or documenting completed training runs

        Returns
        -------
        A markdown-formatted string containing:
            - Model Information section (name, version, iterations, loss metrics)
            - Initial Configuration section (all parameters from loaded state)
            - Training Sessions section (detailed session history with configs)

        Notes
        -----
        The output is suitable for copying into documentation files or displaying in
        terminal interfaces. Each major section is separated by blank lines for readability
        """
        return "\n".join(["## Model",
                          self.render_model_info(), "",
                          self.render_config(), "",
                          "## Sessions",
                          self.render_sessions()])


class Backup:
    """ Creates backup copies of model checkpoints when average loss improves between saves

    This utility class monitors training progress and creates automatic backups of model files
    whenever the current average loss is better (lower) than the best previously recorded loss.
    This provides recovery points for experiments that show initial improvement but later plateau
    or NaN

    The Backup object tracks the lowest average loss observed across all saves, enabling it to
    determine when a backup-worthy improvement has occurred. When triggered, it first creates
    a backup copy with ".bk" extension before updating the recorded best loss value

    Parameters
    ----------
    state
        A State object containing training metadata including:
            - ``lowest_avg_loss``: Best (minimum) average loss observed so far
            - Other state information for context about current model status

    Notes
    -----
    This class is designed to be called with a loss value and file path. It only creates backups
    when:
        1. Loss is greater than 0 (valid loss measurement)
        2. Loss is lower than previously recorded best loss

    The backup preserves the original checkpoint intact and adds a ".bk" extension to avoid
    overwriting files that may have been created by other processes
    """
    def __init__(self, state: State) -> None:
        logger.debug(parse_class_init(locals()))
        self._state = state

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}(state={self._state!r})"

    def _backup(self, model_path: str) -> None:
        """ Backup the given model save file to "{model_name}.bk"

        Parameters
        ----------
        model_path
            The full path to the model save file to back up
        """
        for file in [f"{os.path.splitext(model_path)[0]}.{x}.bk" for x in ("pth", "ckpt")]:
            if os.path.exists(file):
                logger.debug("[Backup] Removing stale backup: '%s'", file)
                os.remove(file)

        backup_file = model_path + ".bk"
        logger.verbose("[Backup] Backing up: '%s' to '%s'",  # type:ignore[attr-defined]
                       model_path, backup_file)
        copyfile(model_path, backup_file)

    def __call__(self, model_path: str, average_loss: float) -> bool:
        """ Create a backup checkpoint if current loss is better than previous best

        Monitors the validation loss and creates a backup when improvement occurs. This method
        should be called after each save to determine if the recent training progress warrants
        preserving an additional recovery point

        Parameters
        ----------
        model_path
            The full path to the latest model save file that may need backing up
        average_loss
            The current average loss value observed since the last save. Must be a non-negative
            float representing mean loss across training samples

        Returns
        -------
        ``True`` if a backup copy was successfully created, ``False`` otherwise

        Notes
        -----
        A backup is only created when:
            - Loss > 0.0 (ensures valid loss measurement)
            - Loss < lowest_avg_loss (actual improvement over best previously seen loss)

        The method updates the State object's lowest_avg_loss after creating a backup, ensuring
        proper tracking across training sessions
        """
        retval = 0.0 < average_loss < self._state.lowest_avg_loss
        if not retval:
            return retval
        self._backup(model_path)
        logger.debug("[Backup] Updating lowest average loss from: %s, to: %s",
                     self._state.lowest_avg_loss, average_loss)
        self._state.lowest_avg_loss = average_loss
        return retval


class Saver:
    """ Handles model checkpoint saving and metadata writing operations

    This utility class manages the core functionality of saving trained models to disk. It
    serializes both model weights and optional training state data into torch-compatible files,
    and generates markdown documentation files describing the saved model's configuration

    The Saver supports two save modes:
        1. Weights-only saves (``is_checkpoint=False``): Stores only neural network parameters
           without optimizer states or training metadata
        2. Full checkpoint saves (``is_checkpoint=True``): Saves complete model including
           weights, state dicts from all training units, and version information

    When saving checkpoints, it automatically generates a markdown info file documenting the
    model's training history, configuration parameters, session details, and performance metrics

    Parameters
    ----------
    model
        The FaceswapModel object containing:
            - ``state``: Training state with iterations, loss metrics, session history
            - ``checkpoint_path``: Path for saving model files
            - Other model metadata
    saveable_units
        Dictionary mapping unit names to training units that have ``state_dict()`` methods.
        These are included in full checkpoint saves (weights + state) but excluded from
        weights-only saves

    Notes
    -----
    Checkpoint saving writes two files:
        1. Model weights only (``*.pth``): Fast loading for inference or further training
        2. Full checkpoint (``model-name.ckpt``): Complete recovery point with all metadata

    The markdown info file (``*_info.md``) is always written to provide human-readable
    documentation of what was saved, useful for progress tracking and debugging
    """
    def __init__(self, model: FaceswapModel, saveable_units: dict[str, TrainingUnit]) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = model
        self._saveable = saveable_units

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"model={self._model!r}, "
                f"saveable_units={self._saveable!r})")

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
        logger.debug("[Saver] Adding saveable units for checkpoint: %s", list(saveable))
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

    def _remove_stale_save(self, filename: str) -> None:
        """ Remove stale save files when switching .pth <-> .ckpt

        Parameters
        ----------
        filename
            The name of the file that has just been saved
        """
        mod, ext = os.path.splitext(filename)
        del_file = mod + (".ckpt" if ext == ".pth" else ".pth")
        if not os.path.exists(del_file):
            return

        logger.debug("[Saver] removing stale save file: '%s'", del_file)
        os.remove(del_file)

    def __call__(self, folder: str, is_checkpoint: bool) -> None:
        """ Save the model weights, optional training state + model info to disk

        Executes a complete save operation including writing files to disk and generating
        documentation. This method should be called after determining that a save is needed

        Parameters
        ----------
        folder
            The directory where model files will be written
        is_checkpoint
            ``True`` if this is a full checkpoint with training state. ``False`` for weights
            only saves which contain just neural network parameters

        Notes
        -----
        Creates the following files in the specified folder:
            - If not checkpoint: ``model-name.pth`` (weights only)
            - If checkpoint: ``model-name.ckpt`` (full model with state)
            - Always: ``model-name_info.md`` (documentation file)
        If the save is switching from weights only to checkpoint (or vice versa) then the previous
        stale save file is removed
        """
        state_dict = self._get_state_dicts(is_checkpoint=is_checkpoint)
        fname = os.path.join(folder, os.path.basename(self._model.checkpoint_path))
        if not is_checkpoint:
            fname = f"{os.path.splitext(fname)[0]}.pth"
        logger.debug("[Saver] Saving %s: '%s'",
                     "checkpoint" if is_checkpoint else "weights", fname)
        logger.verbose("Saving %s...",  # type:ignore[attr-defined]
                       'checkpoint' if is_checkpoint else 'model')
        torch.save(state_dict, fname)
        self._write_model_info(fname)
        self._remove_stale_save(fname)


class Snapshot:
    """ Creates periodic snapshot folders for model recovery points

    This utility class manages the creation of complete training snapshots including both
    model checkpoints and associated logs. Snapshots serve as intermediate recovery points
    between regular checkpoints, providing detailed documentation at specific iteration milestones

    A snapshot consists of three components:
        1. New checkpoint folder created at current iterations
        2. Copy of tensorboard logs for monitoring progress visualization
        3. Markdown documentation describing the snapshot contents

    The Snapshot class automatically removes any previously existing snapshot folder with
    matching iteration count to prevent stale snapshots from accumulating

    Parameters
    ----------
    model
        The FaceswapModel object containing training state and checkpoint information
    saver
        A Saver instance used for writing the actual checkpoint file into the new snapshot folder

    Notes
    -----
    Snapshots are typically created at regular intervals during training (e.g., every 10k
    iterations) to provide intermediate recovery points. The snapshot folder naming follows
    the pattern: ``{src_folder}_snapshot_{iters}_iters``

    Log files are copied from the model's log directory into the snapshot folder, enabling
    detailed progress monitoring through tensorboard or other visualization tools
    """
    def __init__(self, model: FaceswapModel, saver: Saver) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = model
        self._saver = saver

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"model={self._model!r}, "
                f"saver={self._saver!r})")

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
            logger.debug("[Snapshot] Removing previously existing snapshot folder: '%s'", retval)
            rmtree(retval)
        os.makedirs(retval)
        logger.debug("[Snapshot] Snapshot folder: '%s'", retval)
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
        logger.debug("[Snapshot] Copying logs for snapshot: '%s'", os.path.join(destination, logs))
        copytree(os.path.join(src, logs), os.path.join(destination, logs))

    def __call__(self) -> None:
        """ Create a full .ckpt snapshot + tensorboard logs for the given iterations

        Executes the complete snapshot creation process including:
            1. Creating new snapshot directory at current iteration count
            2. Copying tensorboard log files into the snapshot folder
            3. Saving model checkpoint with all training state
            4. Writing markdown documentation file

        This method is typically called when the save interval triggers a snapshot
        """
        logger.info("[Snapshot] Creating model snapshot...")
        dst = self._get_snapshot_folder()
        self._snapshot_logs(dst)
        logger.debug("[Snapshot] Saving snapshot: '%s'", dst)
        self._saver(dst, True)
        logger.info("[Snapshot] Saved: %s iterations", self._model.state.iterations)


class SaveUnit(TrainingUnit):  # pylint:disable=too-many-instance-attributes
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
        self._save_train_state = save_train_state

        self._backup = Backup(model.state)
        self._do_snapshot = False

        self._saver: Saver  # set in on_load
        self._snapshot: Snapshot  # set in on_load

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = ", ".join(f"{k[1:]}={v!r}" for k, v in self.__dict__.items()
                           if k in ("_model",
                                    "_optimizer",
                                    "_events",
                                    "_average_loss",
                                    "_save_interval",
                                    "_snapshot_interval",
                                    "_save_train_state"))
        return f"{self.__class__.__name__}({params})"

    def on_load(self, loop: TrainStep) -> None:
        """ Initialize the model saver and configure saving behavior

        Stores references to all training units that can provide state dictionaries for inclusion
        in saved checkpoints

        Parameters
        ----------
        loop
            The training step object managing this unit's lifecycle
        """
        self._saver = Saver(self._model, loop.units.have_state_dict)
        self._snapshot = Snapshot(self._model, self._saver)
        logger.debug("%s Created. saver: %s, snapshot: %s",
                     self.log_name, self._saver, self._snapshot)

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

    def _backup_and_save(self, is_exit: bool) -> None:
        """ Execute the complete save operation for either normal saves or exit

        Parameters
        ----------
        is_exit
            Whether this save is happening at training exit
        """
        average_loss = self._get_average_loss()
        is_checkpoint = self._save_train_state == "always" or (is_exit and
                                                               self._save_train_state == "exit")

        print("\x1b[2K", end="\r")  # Clear last line for line length (verbose/info coming soon)
        self._saver(os.path.dirname(self._model.checkpoint_path), is_checkpoint)

        latest_save = self._model.latest_save
        assert latest_save is not None
        has_backup = self._backup(latest_save, average_loss)

        msg = f"[Saved {'checkpoint' if is_checkpoint else 'model'}]"
        if average_loss != 0.0:
            msg += f" - Average loss since save: {average_loss:.5f}"
        if has_backup:
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
        self._backup_and_save(False)
        if self._do_snapshot:
            print("\x1b[2K", end="\r")  # Clear last line for line length (verbose/info coming)
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

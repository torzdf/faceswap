#!/usr/bin/env/python3
""" Manages persistent model training state across sessions and checkpoints

This module provides the core state management infrastructure for Faceswap's deep learning
training pipeline. It handles serialization of configuration parameters and training metadata to
disk while maintaining in-memory tracking of progress between saves.

Key Components:
    1. _Config: Configuration manager for hyperparameters (learning rate, batch size, etc.)
       Separates fixed (architecture-dependent) from updatable (user-adjustable) settings

    2. Session: Dataclass recording individual training phases with metadata including
       start timestamp, iteration counts, and session-specific configuration

    3. State: Main container object tracking:
        - Plugin version compatibility
        - Total iterations across all sessions
        - Lowest validation loss observed
        - Learning rate finder results
        - Complete history of completed training sessions

Workflow Overview:
------------------
1. Initialize State with plugin path during model creation
2. Call set_pre_training() after loading checkpoint or before first session
3. Call set_training() once initial setup (LRFinder, validation) is complete
4. increment_iterations() for each training batch processed
5. add_new_session() when hyperparameters change significantly (e.g., new learning rate)
6. state_dict() called periodically to save checkpoints with all metadata

The State object enables:
    - Resuming interrupted training from last checkpoint
    - Tracking convergence metrics across multiple sessions
    - Preserving optimal learning rates discovered during LRFinder scans
    - Maintaining version compatibility for model architecture upgrades

Additional Utilities:
---------------------
- StateMarkdown: Converts state objects into formatted markdown tables for saving to file. Provides
methods to render summary tables, configuration parameters, and session history.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import logging
import time
import typing as T
from importlib import import_module
from inspect import isclass

from lib.config.objects import ConfigItem, GlobalSection
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train import train_config as cfg

logger = logging.getLogger(__name__)


_VERSION = 1.0
""" The current state_dict version for serialization compatibility across Faceswap releases """


class _Config:
    """ Manages the updatable configuration items for Faceswap model training sessions

    This utility class handles loading, storing, and updating configuration parameters that control
    model behavior during training. It separates fixed (read-only) settings from updatable ones,
    allowing runtime adjustments to hyperparameters without modifying saved state

    Configuration is loaded from two sources:
        1. Global config options defined in train_config module
        2. Model-specific defaults found at ``{plugin_path}_defaults`` module

    All values are callable objects (ConfigItem) that return their current value when called,
    enabling stateless access to configuration parameters. When loading from a checkpoint, existing
    values are updated if different and the parameter is not marked as fixed

    Parameters
    ----------
    plugin_path
        The relative import path string to the model plugin's module from the faceswap root
        (e.g., "lib.model.plugin.networks.resnet"). This is used to load model-specific
        configuration defaults from ``{plugin_path}_defaults`` module during initialization

    Notes
    -----
    Fixed config items retain their values and cannot be updated by loaded state. Updatable
    items are replaced with values currently set in Faceswap's config when loading, allowing users
    to fine-tune specific hyperparameters for particular training runs while preserving learned
    settings from previous sessions

    The class generates initial configuration by combining global options (training parameters)
    and model-specific defaults (architecture-dependent settings). This separation allows both
    general-purpose configurations and plugin-specific optimizations to coexist in the same state
    """

    def __init__(self, plugin_path: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._import_path = plugin_path
        self._name = f"[_Config.{plugin_path.rsplit(".", maxsplit=1)[-1]}]"
        self._config, self._updatable = self._generate_config()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}(plugin_path={repr(self._import_path)})"

    @property
    def config(self) -> dict[str, T.Any]:
        """ Dictionary of configuration parameters where keys are parameter names and values are
        their current settings as returned by each ConfigItem's call method """
        return {k: v() for k, v in self._config.items()}

    @property
    def session_config(self) -> dict[str, T.Any]:
        """ Dictionary of updatable config items with their current values for this session """
        return {k: v() for k, v in self._config.items() if k in self._updatable}

    def _get_global_options(self) -> dict[str, ConfigItem]:
        """ Obtain all of the current global user config options

        Returns
        -------
        All of the current global user configuration options
        """
        objects = {key: val for key, val in vars(cfg).items()
                   if isinstance(val, ConfigItem)
                   or isclass(val) and issubclass(val, GlobalSection) and val != GlobalSection}

        retval: dict[str, ConfigItem] = {}
        for key, obj in objects.items():
            if isinstance(obj, ConfigItem):
                retval[key] = obj
                continue
            for name, opt in obj.__dict__.items():
                if isinstance(opt, ConfigItem):
                    retval[name] = opt
        logger.debug("%s Loaded global config options: %s",
                     self._name, {k: v.value for k, v in retval.items()})
        return retval

    def _get_model_options(self) -> dict[str, ConfigItem]:
        """ Obtain all of the currently configured model user config options

        Returns
        -------
        The currently configured model plugin options
        """
        mod_name = f"{self._import_path}_defaults"
        try:
            mod = import_module(mod_name)
        except ModuleNotFoundError:
            logger.debug("%s No plugin specific defaults file found at '%s'",
                         self._name, mod_name)
            return {}

        retval = {k: v for k, v in vars(mod).items() if isinstance(v, ConfigItem)}
        logger.debug("%s Loaded plugin config options: %s",
                     self._name, {k: v.value for k, v in retval.items()})
        return retval

    def _generate_config(self) -> tuple[dict[str, ConfigItem], list[str]]:
        """ Generate initial state configuration by merging global and model-specific options

        Returns
        -------
        config
            Dictionary mapping all configuration parameter names to their ConfigItem objects
        updatable
            List of updatable item names (parameters that can be modified during training)
        """
        config: dict[str, ConfigItem] = {}
        updatable: list[str] = []
        options = self._get_global_options() | self._get_model_options()
        for key, val in options.items():
            config[key] = val
            if not val.fixed:
                updatable.append(key)

        logger.debug("%s Generated initial state config: %s", self._name, config)
        logger.debug("%s Updatable items: %s", self._name, updatable)
        return config, updatable

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """ Load configuration parameters from a saved state dictionary into this object

        Updates all fixed configuration items with values from the provided state_dict.
        Fixed parameters are preserved from the state_dict and not overwritten by their current
        values in Faceswap config. When values differ for updatable parameters, an info log message
        is recorded showing the old-to-new value transition for transparency

        Parameters
        ----------
        state_dict
            Dictionary containing configuration parameter names as keys and their saved
            values from a checkpoint file. Each value should be compatible with the type
            of its corresponding ConfigItem object
        """
        for key, opt in self._config.items():
            val = opt()

            if key not in state_dict:
                logger.info("Adding new config item to state file: '%s': %s", key, repr(val))
                continue

            loaded_val = state_dict[key]

            if not opt.fixed and val != loaded_val:
                logger.info("Config item: '%s' has been updated from %s to %s",
                            key, repr(loaded_val), repr(val))
                continue

            if val != loaded_val:
                logger.debug("%s Fixed config item '%s' Updated from %s to %s from state file",
                             self._name, key, repr(val), repr(loaded_val))
                opt.set(loaded_val)
        logger.info("Using configuration saved in state file")
        logger.debug("%s Loaded state_dict: %s", self._name, state_dict)

    def state_dict(self) -> dict[str, T.Any]:
        """ Serialize all current configuration values to a dictionary for checkpoint saving

        Returns a dictionary mapping each configuration parameter name to its current value.
        Each value is obtained by calling the corresponding ConfigItem callable object.
        Also includes version metadata for compatibility checks when loading from future
        checkpoints

        Returns
        -------
        Dictionary containing all config parameters with their values and a ``version`` key
        set to the current state version for serialization format identification
        """
        conf = {k: v() for k, v in self._config.items()}
        return conf | {"version": _VERSION}


@dataclass
class Session:
    """ Records training session metadata and statistics for persistence

    This dataclass holds information about individual model training sessions including
    batch size configuration, start timestamp, iteration counts, and associated user config.
    Sessions are used to track different training phases (e.g., initial learning rate
    finding, main training with different hyperparameters) within the same model checkpoint

    Each session stores:
        1. ``batch_size``: Number of samples per training iteration
        2. ``config``: User-adjustable parameters for this specific run
        3. ``timestamp``: When the session started
        4. ``iterations``: How many iterations this session has processed

    Parameters
    ----------
    batch_size
        The number of training samples processed per side per iteration during this session
    config
        Updatable configuration items specific to this training session (learning rate, etc.)
    timestamp
        Unix timestamp marking when this session began
    iterations, optional
        Count of completed training iterations within this session. Default: 0
    """
    batch_size: int
    """ Samples per training iteration for this session """
    config: dict[str, T.Any]
    """ Session-specific adjustable parameters as a dictionary """
    timestamp: float = time.time()
    """ Start time of the session in Unix epoch seconds """
    iterations: int = 0
    """ Total iterations processed during this session's lifetime """


class State:  # pylint:disable=too-many-instance-attributes
    """ Container class for managing model configuration and training state and session information

    State holds all persistent metadata about a model's training lifecycle including version
    tracking, iteration counts, lowest loss values, learning rate finder results, and complete
    history of training sessions. This enables resuming interrupted training while preserving
    progress and hyperparameter settings, as well as persisting a Faceswap model's configuration

    The class manages three main categories:
        1. Version and metadata (plugin_version, iterations, lr_finder)
        2. Training metrics (lowest_avg_loss from validation curves)
        3. Session data (all training phases with their configs)

    Parameters
    ----------
    plugin_path
        Relative import path string to the model plugin module (e.g.
        "lib.model.plugin.networks.resnet"). Used to identify the plugin and load model-specific
        configuration defaults
    """
    def __init__(self, plugin_path: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._repr_obj = (f"{self.__class__.__name__}(plugin_path={repr(plugin_path)})")

        self.plugin_name = plugin_path.rsplit(".")[-1].replace("_", "-")
        """ The model name with underscores converted to dashes for user-friendly display """

        self.lr_finder = -1.0
        """ Learning rate found via LRFinder for optimal training (or -1 if not discovered) """

        self.lowest_avg_loss: float = 0.0
        """ Minimum average loss observed between checkpoint saves """

        self._sessions: dict[int, Session] = {}
        self._plugin_version = 0.0
        self._config = _Config(plugin_path)
        self._total_steps = 0
        self._step_called = False

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return self._repr_obj

    @property
    def config(self) -> dict[str, T.Any]:
        """ All current configuration values (fixed and updatable combined) """
        return self._config.config

    @property
    def plugin_version(self) -> float:
        """ The version of the Faceswap model architecture currently in use """
        assert self._plugin_version, "Plugin version has not been set"
        return self._plugin_version

    @property
    def session_config(self) -> dict[str, T.Any]:
        """ Dictionary of current session's adjustable parameters (learning rate, etc.) """
        return self._sessions[self.session_id].config

    @property
    def session_id(self) -> int:
        """ The ID number identifying the current active training session

        Returns 0 if no sessions exist or in pre-train mode. Otherwise returns the maximum
        session ID, which corresponds to the most recently created and currently running phase.
        Session IDs are auto-incremented starting from 1 when new sessions begin """
        if not self._sessions:
            return 0
        return max(self._sessions)

    @property
    def iterations(self) -> int:
        """ Total training iterations across all sessions """
        return self._total_steps

    @property
    def session_iterations(self) -> int:
        """ Iterations processed during the current active training session only

        If no sessions exist or we're in pre-train mode, returns 0. Otherwise returns the
        iteration count stored in the current session object """
        if not self._sessions or self.session_id not in self._sessions:
            return 0
        return self._sessions[self.session_id].iterations

    def set_plugin_version(self, version: float) -> None:
        """ Set the plugin version for a newly initialized Faceswap model

        Parameters
        ----------
        version
            The version of the plugin that has been loaded
        """
        logger.debug("[State] Setting plugin_version: %s", version)
        self._plugin_version = version

    def load_state_dict(self, state_dict: dict[str, T.Any]):
        """ Restore all state from a saved checkpoint dictionary

        Loads complete model state including plugin version, iteration counts, loss metrics,
        learning rate finder results, and all previous training sessions. Configuration items
        are loaded from the config section while session data is reconstructed from the
        sessions dictionary. Pre-train mode flag (-1 iterations) is also restored

        Parameters
        ----------
        state_dict
            Complete checkpoint dictionary containing:
            - ``plugin_version``: Model architecture version number
            - ``iterations``: Total training steps completed
            - ``lowest_avg_loss``: Best validation loss observed
            - ``lr_finder``: Optimal learning rate (or -1.0)
            - ``sessions``: List of past session dictionaries with metadata
            - ``config``: Current configuration parameters
            - ``version``: Serialization format version

        Notes
        -----
        The state_dict should be obtained from calling FaceswapModel.state_dict(). This method
        is symmetric to that function and enables full restoration of training state across
        sessions or after interruption
        """
        self._plugin_version = state_dict.get("plugin_version", 0.0)
        self._total_steps = state_dict.get("iterations", 0)
        self.lowest_avg_loss = state_dict.get("lowest_avg_loss", 0.0)
        self.lr_finder: float = state_dict.get("lr_finder", -1.0)
        self._sessions = {k: Session(**v) for k, v in state_dict.get("sessions", {}).items()}
        self._config.load_state_dict(state_dict.get("config", {}))
        logger.debug("[State] Loaded state_dict: %s", state_dict)

    def state_dict(self) -> dict[str, T.Any]:
        """ Serialize complete model training state to a checkpoint-compatible dictionary

        Returns a comprehensive dictionary containing all persistent state information including:
            - ``plugin_name``: Model identifier with dashes for user-facing display
            - ``plugin_version``: Neural network architecture version number
            - ``iterations``: Total training steps across all sessions (or -1 in pre-train mode)
            - ``lowest_avg_loss``: Best validation loss observed between saves
            - ``lr_finder``: Optimal learning rate discovered (-1 if not run yet)
            - ``sessions``: All completed session data with batch sizes and configs
            - ``config``: Current hyperparameter settings (fixed and updatable)
            - ``version``: Serialization format version for compatibility checks

        This dictionary is what gets saved to checkpoint files (model.ckpt/model.pth). When
        loading, calling load_state_dict() will restore the model to exactly where training left
        off

        Returns
        -------
        Dictionary ready for serialization with all training metadata and metrics
        """
        return {"plugin_name": self.plugin_name,
                "plugin_version": self.plugin_version,
                "iterations": self._total_steps,
                "lowest_avg_loss": self.lowest_avg_loss,
                "lr_finder": self.lr_finder,
                "sessions": {k: asdict(v) for k, v in self._sessions.items()
                             if v.iterations > 0},
                "config": self._config.state_dict(),
                "version": _VERSION}

    def add_new_session(self, batch_size: int) -> None:
        """ Begin a new training session with specified hyperparameters

        Creates a fresh Session object with the given batch size and copies current updatable
        config values (learning rate, optimizer settings, etc.)

        Each session gets auto-incremented ID based on existing sessions. The new session starts
        fresh with iteration count 0 and current timestamp. Fixed config items are not copied -
        only updatable ones become part of the new session's configuration

        Parameters
        ----------
        batch_size
            Number of training samples processed per side per iteration for this new session
        """
        session_id = self.session_id + 1
        config = self._config.session_config
        self._sessions[session_id] = Session(batch_size, config)
        logger.debug("[State] Created training session %s: batch_size: %s, session: %s",
                     session_id, batch_size, self._sessions[session_id])

    def increment_iterations(self) -> None:
        """ Advance training step counters for both total sessions and current phase

        Increments both the cumulative iteration count across all training sessions and the
        iteration counter within the currently active session. If in pre-train mode (total
        steps == -1), no increments occur since this flag indicates initialization phase where
        iteration counting is disabled until proper training begins

        Notes
        -----
        This method should be called once per training batch/iteration. In pre-train mode,
        we prevent incrementing to maintain correct state semantics. When exiting  pre-train mode
        via set_training(), counters reset and normal iteration tracking resumes
        """
        if self._total_steps < 0:
            logger.trace(  # type:ignore[attr-defined]
                "[State] In pre-train mode. Not incrementing"
                )
            return
        self._total_steps += 1
        self._sessions[self.session_id].iterations += 1

    def set_pre_training(self) -> None:
        """ Mark the model as being in initialization/pre-training phase

        Sets total step counter to -1, indicating that iteration counting is disabled. This
        state is used before actual training begins and after loading checkpoints where no
        iterations have been recorded yet. In this mode, increment_iterations() will not count
        steps since pre-train mode signifies the model hasn't started proper training loops

        Raises
        ------
        AssertionError
            If training has already commenced (iterations > 0). This prevents incorrectly re-
            entering pre-training state after normal operation

        Notes
        -----
        Called when loading a fresh checkpoint without saved iterations, or before starting
        initial training sessions. The assertion ensures this is only done on genuinely new models
        that haven't completed any training rounds
        """
        logger.debug("[State] Entering pre-train mode")
        assert self._total_steps <= 0, "Pre-train mode can only be called on new models."
        self._total_steps = -1

    def set_training(self) -> None:
        """ Transition from pre-training to active training state

        Resets the total step counter from -1 (pre-train flag) to 0 and enables iteration counting.
        This marks that the model is now in proper training mode where increment_iterations() will
        begin recording actual progress toward convergence targets

        Raises
        ------
        AssertionError
            If called when not already in pre-training mode (total_steps != -1). Ensures state
            transitions are sequential: fresh load → set_pre_training → set_training → training

        Notes
        -----
        Typically called after learning rate finding or initial validation before starting the
        main training loop. The assertion enforces proper lifecycle management to prevent skipping
        initialization phases that track session boundaries correctly
        """
        logger.debug("[State] Entering train mode")
        assert self._total_steps == -1, "Train mode can only be called when in pre-train mode."
        self._total_steps = 0


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
        - ``render_summary()``: Generates detailed tables for each training session with batch
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
            "lowest_avg_loss": f"{state['lowest_avg_loss']:.8f}",
            "lr_finder": "N/A" if state["lr_finder"] < 0 else state["lr_finder"],
            "created": self._format_time(state["sessions"].get(1, {}).get("timestamp",
                                                                          time.time()))
        }
        data = {k.replace("plugin_", "").replace("_", " ").title(): str(v)
                for k, v in data.items()}
        return "\n".join(["## Model Information"] + self._format_to_table(data))

    def render_config(self) -> str:
        """ Format all configuration parameters as a markdown table

        Returns a detailed table showing every configuration parameter and its value set at the
        initial state

        Returns
        -------
        A markdown-formatted string containing a sorted list of all configuration parameters
        as a two-column table (Parameter | Value)

        Notes
        -----
        Configuration parameters are shown in their original state from the checkpoint, not
        current values that may have been modified since loading. Useful for reference when
        reviewing saved training runs
        """
        state = self._state.state_dict()
        data = {"Parameter": list(state["config"]),
                "Value": [f"{x!r}" for x in state["config"].values()]}
        return "\n".join(["## Initial Config"] + self._format_to_table(data))

    def render_summary(self) -> str:
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

        lines = ["## Sessions"]

        for idx, (session, conf) in enumerate(zip(reversed(info), reversed(conf))):
            lines.append(f"### Session {len(state["sessions"]) - idx}")
            lines.extend(self._format_to_table(session) + [""])
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
        return "\n".join([self.render_model_info(), "",
                          self.render_config(), "",
                          self.render_summary()])


__all__ = get_module_objects(__name__)

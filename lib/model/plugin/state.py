#!/usr/bin/env python3
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
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import logging
import time
import typing as T

from lib.config.objects import ConfigItem
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train.train_config import load_config

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
    plugin_name
        The standardized internal name of the plugin that is being loaded

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
    def __init__(self, plugin_name: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._plugin_name = plugin_name
        self._log_name = f"[_Config.{plugin_name.rsplit(".", maxsplit=1)[-1]}]"
        self._config, self._updatable = self._generate_config()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}(plugin_name={self._plugin_name!r})"

    @property
    def config(self) -> dict[str, T.Any]:
        """ Dictionary of configuration parameters where keys are parameter names and values are
        their current settings as returned by each ConfigItem's call method """
        return {k: v() for k, v in self._config.items()}

    @property
    def session_config(self) -> dict[str, T.Any]:
        """ Dictionary of updatable config items with their current values for this session """
        return {k: v() for k, v in self._config.items() if k in self._updatable}

    @property
    def fixed_config(self) -> dict[str, T.Any]:
        """ Dictionary of config items that are fixed on model creation """
        return {k: v() for k, v in self._config.items() if k not in self._updatable}

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

        opts = load_config().sections
        global_opts = {k: v for name, sect in opts.items()
                       for k, v in sect.options.items()
                       if not name.startswith(("model.", "trainer."))}
        local_opts = {k: v for name, sect in opts.items()
                      for k, v in sect.options.items()
                      if name == f"model.{self._plugin_name}"}

        for key, val in (global_opts | local_opts).items():
            config[key] = val
            if not val.fixed:
                updatable.append(key)

        logger.debug("%s Generated initial state config: %s", self._log_name, config)
        logger.debug("%s Updatable items: %s", self._log_name, updatable)
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
                             self._log_name, key, repr(val), repr(loaded_val))
                opt.set(loaded_val)
        logger.info("Using configuration saved in state file")
        logger.debug("%s Loaded state_dict: %s", self._log_name, state_dict)

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
    plugin_name
        The standardized internal name of the plugin that is being loaded
    """
    def __init__(self, plugin_name: str) -> None:
        logger.debug(parse_class_init(locals()))
        self.plugin_name = plugin_name.replace("_", "-")
        """ The model name with underscores converted to dashes for user-friendly display """

        self.lr_finder = -1.0
        """ Learning rate found via LRFinder for optimal training (or -1 if not discovered) """

        self.lowest_avg_loss: float = 0.0
        """ Minimum average loss observed between checkpoint saves """

        self._sessions: dict[int, Session] = {}
        self._plugin_version = 0.0
        self._config = _Config(plugin_name)
        self._total_steps = 0
        self._step_called = False

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}(plugin_name={self.plugin_name.replace('-', '_')!r})")

    @property
    def config(self) -> dict[str, T.Any]:
        """ All current configuration values (fixed and updatable combined) """
        return self._config.config

    @property
    def fixed_config(self) -> dict[str, T.Any]:
        """ All configuration values that are fixed on model creation """
        return self._config.fixed_config

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


__all__ = get_module_objects(__name__)

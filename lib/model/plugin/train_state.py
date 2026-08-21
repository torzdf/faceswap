#!/usr/bin/env python3
""" Tensorboard call back for PyTorch logging. Hopefully temporary until a native Keras version
is implemented """
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
""" The current state_dict version """


class _Config:
    """ Manages the updatable config items

    Parameters
    ----------
    plugin_path: str
        The relative import path to the model plugin's module from the faceswap root
    """
    def __init__(self, plugin_path: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._import_path = plugin_path
        self._name = f"[_Config.{plugin_path.rsplit(".", maxsplit=1)[-1]}]"
        self._config, self._updatable = self._generate_config()

    def __repr__(self) -> str:
        """ Cleaner logging """
        return f"{self.__class__.__name__}(plugin_path={repr(self._import_path)})"

    @property
    def config(self) -> dict[str, T.Any]:
        """ The currently set values for all config items """
        return {k: v() for k, v in self._config.items()}

    @property
    def session_config(self) -> dict[str, T.Any]:
        """ The currently set values for any updatable config items """
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
        """ Generate an initial state config based on the currently selected user config

        Returns
        -------
        config
            The currently selected global and model specific config options
        updatable
            Config item names that can be adjusted for a loaded model
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
        """ Load the contents of the state_dict into this state object

        Parameters
        ----------
        state_dict
            The _Config state_dict for the running Faceswap model
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
        """ This _config object's state_dict """
        conf = {k: v() for k, v in self._config.items()}
        return conf | {"version": 1.0}


@dataclass
class Session:
    """ Holds information about training sessions """
    batch_size: int
    """ Batch size for the session """
    config: dict[str, T.Any]
    """ Updatable config items for the session """
    timestamp: float = time.time()
    """ Session start time stamps """
    iterations: int = 0
    """ Number of iterations processed for the session """


class State:  # pylint:disable=too-many-instance-attributes
    """ Holds information about the training state of the model

    Parameters
    ----------
    plugin_path
        The relative import path to the model plugin's module from the faceswap root
    """
    def __init__(self, plugin_path: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._repr_obj = (f"{self.__class__.__name__}(plugin_path={repr(plugin_path)})")

        self.plugin_name = plugin_path.rsplit(".")[-1].replace("_", "-")
        self.lr_finder = -1.0
        """ The value discovered from the learning rate finder. -1 if no value stored """
        self._sessions: dict[int, Session] = {}
        self.lowest_avg_loss: float = 0.0
        """ float: The lowest average loss seen between save intervals. """
        self._plugin_version = 0.0
        self._config = _Config(plugin_path)
        self._total_steps = 0
        self._step_called = False

    def __repr__(self) -> str:
        """ Cleaner logging """
        return self._repr_obj

    @property
    def config(self) -> dict[str, T.Any]:
        """ The currently set values for all config items """
        return self._config.config

    @property
    def plugin_version(self) -> float:
        """ The version of the plugin that this state file corresponds to in use """
        assert self._plugin_version, "Plugin version has not been set"
        return self._plugin_version

    @property
    def session_config(self) -> dict[str, T.Any]:
        """ The current session config as it will get serialized to disk """
        return self._sessions[self.session_id].config

    @property
    def session_id(self) -> int:
        """ The current session ID. If training has not yet commenced, this will be the last
        session ID trained. If the first training step has been reached, this will be the currently
        training session ID """
        if not self._sessions:
            return 0
        return max(self._sessions)

    @property
    def iterations(self) -> int:
        """ The total number of iterations the model has been trained for """
        return self._total_steps

    @property
    def session_iterations(self) -> int:
        """ The number of iterations the model has been trained for during the current session """
        if not self._sessions or self.session_id not in self._sessions:
            return 0
        return self._sessions[self.session_id].iterations

    def set_plugin_version(self, version: float) -> None:
        """ Set the plugin version for a newly initialized model

        Parameters
        ----------
        version
            The version of the plugin that has been loaded
        """
        logger.debug("[State] Setting plugin_version: %s", version)
        self._plugin_version = version

    def load_state_dict(self, state_dict: dict[str, T.Any]):
        """ Load the contents of the state_dict into this state object

        Parameters
        ----------
        state_dict
            The State state dict for the running Faceswap model
        """
        self._plugin_version = state_dict.get("plugin_version", 0.0)
        self._total_steps = state_dict.get("iterations", 0)
        self.lowest_avg_loss = state_dict.get("lowest_avg_loss", 0.0)
        self.lr_finder: float = state_dict.get("lr_finder", -1.0)
        self._sessions = {k: Session(**v) for k, v in state_dict.get("sessions", {}).items()}
        self._config.load_state_dict(state_dict.get("config", {}))
        logger.debug("[State] Loaded state_dict: %s", state_dict)

    def state_dict(self) -> dict[str, T.Any]:
        """ This State object's state_dict """
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
        """ Add a new training session with the specified batch size

        Parameters
        ----------
        batch_size
            The number of faces to process per training iteration for this session
        """
        session_id = self.session_id + 1
        config = self._config.session_config
        self._sessions[session_id] = Session(batch_size, config)
        logger.debug("[State] Created training session %s: batch_size: %s, session: %s",
                     session_id, batch_size, self._sessions[session_id])

    def increment_iterations(self) -> None:
        """ Increment the session and total iterations by 1 """
        if self._total_steps < 0:
            logger.trace(  # type:ignore[attr-defined]
                "[State] In pre-train mode. Not incrementing"
                )
            return
        self._total_steps += 1
        self._sessions[self.session_id].iterations += 1

    def set_pre_training(self) -> None:
        """ Set the state object to pre-train mode """
        logger.debug("[State] Entering pre-train mode")
        assert self._total_steps <= 0, "Pre-train mode can only be called on new models."
        self._total_steps = -1

    def set_training(self) -> None:
        """ Set the state object to train mode """
        logger.debug("[State] Entering train mode")
        assert self._total_steps == -1, "Train mode can only be called when in pre-train mode."
        self._total_steps = 0


__all__ = get_module_objects(__name__)

#!/usr/bin/env python3
"""Tensorboard call back for PyTorch logging. Hopefully temporary until a native Keras version
is implemented"""
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


class _Config:
    """Manages the updatable config items

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
        """Cleaner logging"""
        return f"{self.__class__.__name__}(plugin_path={repr(self._import_path)})"

    @property
    def session_config(self) -> dict[str, T.Any]:
        """The currently set values for any updatable config items"""
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
        """Load the contents of the state_dict into this state object

        Parameters
        ----------
        state_dict
            The _Config state_dict for the running Faceswap model
        """
        # TODO move old legacy code in _base.state to migration
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
        """This _config object's state_dict"""
        return {k: v() for k, v in self._config.items()}


@dataclass
class Session:
    """Holds information about training sessions"""
    batch_size: int
    """Batch size for the session"""
    config: dict[str, T.Any]
    """Updatable config items for the session"""
    timestamp: float = time.time()
    """Session start time stamps"""
    iterations: int = 0
    """Number of iterations processed for the session"""


class State:
    """Holds information about the training state of the model

    Parameters
    ----------
    plugin_path
        The relative import path to the model plugin's module from the faceswap root
    batch_size
        The batch size that the model is to be trained at, if opening for a training session,
        otherwise ``None``. Default: ``None``
    """
    def __init__(self, plugin_path: str, batch_size: int | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._repr = (f"{self.__class__.__name__}(plugin_path={repr(plugin_path)}, "
                      f"batch_size={batch_size}")

        self._batch_size = batch_size
        self.lr_finder = -1.0
        """The value discovered from the learning rate finder. -1 if no value stored"""
        self._sessions: dict[int, Session] = {}
        self.lowest_avg_loss: float = 0.0
        """float: The lowest average loss seen between save intervals. """
        self.learning_rate_from_finder: bool = False
        """bool. Set to ``True`` if learning rate is being read from the finder rather than user
        config"""
        self._config = _Config(plugin_path)
        self._version = 2.0

        self._total_steps = 0
        self._step_called = False

    def __repr__(self) -> str:
        """Cleaner logging"""
        return self._repr

    @property
    def session_id(self) -> int:
        """The current session ID. If training has not yet commenced, this will be the last session
        ID trained. If the first training step has been reached, this will be the currently
        training session ID"""
        if not self._sessions:
            return 0
        return max(self._sessions)

    @property
    def iterations(self) -> int:
        """The total number of iterations the model has been trained for"""
        return self._total_steps

    @property
    def session_iterations(self) -> int:
        """The number of iterations the model has been trained for during the current session"""
        if not self._sessions or self.session_id not in self._sessions:
            return 0
        return self._sessions[self.session_id].iterations

    def load_state_dict(self, state_dict: dict[str, T.Any]):
        """Load the contents of the state_dict into this state object

        Parameters
        ----------
        state_dict
            The State state dict for the running Faceswap model
        """
        self._total_steps = state_dict.get("iterations", 0)
        self._sessions = {k: Session(**v) for k, v in state_dict.get("sessions", {}).items()}
        self.lowest_avg_loss = state_dict.get("lowest_avg_loss", 0.0)
        self.lr_finder = state_dict.get("lr_finder", -1.0)
        self._config.load_state_dict(state_dict.get("config", {}))
        logger.debug("[State] Loaded state_dict: %s", state_dict)

    def state_dict(self) -> dict[str, T.Any]:
        """This State object's state_dict"""
        return {"iterations": self._total_steps,
                "lowest_avg_loss": self.lowest_avg_loss,
                "lr_finder": self.lr_finder,
                "sessions": {k: asdict(v) for k, v in self._sessions.items()
                             if v.iterations > 0},
                "config": self._config.state_dict(),
                "version": self._version}

    def step(self) -> None:
        """Increment the session and total steps

        Parameters
        ----------
        lr_from_finder
            ``True`` if the learning rate
        """
        if not self._step_called:
            assert self._batch_size is not None, "batch_size must be provided when training"
            config = self._config.session_config
            if self.learning_rate_from_finder:  # TODO check this
                logger.debug("[State] Storing learning rate from finder: %s", self.lr_finder)
                config["learning_rate"] = self.lr_finder
            self._sessions[self.session_id + 1] = Session(self._batch_size, config)
            self._step_called = True

        self._total_steps += 1
        self._sessions[self.session_id].iterations += 1


__all__ = get_module_objects(__name__)

#!/usr/bin/env python3
"""Wraps the selected Torch optimizer and handles optimizer related functions such as loss scaling,
clipping and gradient accumulation"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.autoclip import AutoClipper
from lib.model import optimizers
from lib.utils import get_module_objects

from .lr_warmup import WarmupScheduler
from .lr_finder import LRFScheduler

if T.TYPE_CHECKING:
    from keras import Variable
    from plugins.train.model.base import ModelPlugin
    from plugins.train.train_config import Optimizer as OptConfig


logger = logging.getLogger(__name__)

_OPTIMIZERS = {"adabelief": optimizers.AdaBelief,
               "adam": torch.optim.Adam,
               "adamax": torch.optim.Adamax,
               "adamw": torch.optim.AdamW,
               "lion": optimizers.Lion,
               "nadam": torch.optim.NAdam,
               "rms-prop": torch.optim.RMSprop}


# TODO keep for legacy weights update
def get_parameter_group_ids(trainable_variables: list[Variable]
                            ) -> dict[int, T.Literal["decay", "no_decay"]]:
    """Obtain the index of each item in the keras model's trainable weights that belong to each
    of the optimizer's parameter groups (ie split by weights that take decay and don't take decay)

    Parameters
    ----------
    trainable_variables
        list of trainable variables from keras model

    Returns
    -------
    dictionary of keras model's trainable weight index to the name of the parameter group
    """
    retval: dict[int, T.Literal["decay", "no_decay"]] = {}
    for idx, var in enumerate(trainable_variables):
        retval[idx] = "no_decay" if var.ndim <= 1 or var.name.endswith("bias") else "decay"

    logger.debug("parameter group ids: %s", retval)
    return retval


class GradClip:
    """Handles the clipping of gradients based on user supplied parameters

    Parameters
    ----------
    method
        The clipping method to use
    value
        The clipping value to use. For autoclip this is the percentile to clip at (a value of 1.0
        will clip at the 10th percentile a value of 2.5 will clip at the 25th percentile etc)
    autoclip_history
        The history length for auto clipping. Default: 10000
    """
    def __init__(self,
                 method: T.Literal["autoclip", "global_norm", "norm", "value"],
                 value: float,
                 autoclip_history: int = 10000) -> None:
        logger.debug(parse_class_init(locals()))
        self._value = value
        self._clipper = self._get_clipper(method, autoclip_history)

    @classmethod
    def _clip_norm(cls, parameters: list[nn.Parameter], max_norm: float) -> None:
        """Clip each parameter independently by its own norm

        Parameters
        ----------
        parameters
            The parameters to clip
        max_norm
            The value to clip by
        """
        with torch.no_grad():
            for param in parameters:
                if param.grad is None:
                    continue
                grad = param.grad
                norm = grad.norm(2)
                if norm > max_norm:
                    grad.mul_(max_norm / norm)

    def _get_clipper(self,
                     method: T.Literal["autoclip", "global_norm", "norm", "value"],
                     autoclip_history: int) -> T.Callable[[list[nn.Parameter], float],
                                                          None | torch.Tensor]:
        """Obtain the correct function to clip the gradients based on the selected method

        Parameters
        ----------
        method
            The clipping method to use
        autoclip_history
            The history length for auto clipping

        Returns
        -------
        The function used to clip the gradients
        """
        methods: dict[str, T.Callable[[list[nn.Parameter], float], None | torch.Tensor]] = {
            "autoclip": AutoClipper(int(self._value * 10), history_size=autoclip_history),
            "global_norm": nn.utils.clip_grad_norm_,
            "norm": self._clip_norm,
            "value": nn.utils.clip_grad_value_}
        if method not in methods:
            raise ValueError(f"'{method}' is not a valid clipping method. Select "
                             f"from {list(methods)}")
        retval = methods[method]
        logger.debug("[GradClip] Got clipper '%s': %s", method, retval)
        return retval

    def __call__(self, parameters: list[nn.Parameter]) -> None:
        """Clip the given parameters by the chosen method

        Parameters
        ----------
        parameters
            The parameters to clip
        """
        self._clipper(parameters, self._value)


class Optimizer:
    """Object for managing the selected Torch optimizer

    Parameters
    ----------
    model
        The model that is to be trained
    config
        The optimizer user configuration options
    mixed_precision
        ``True`` to train using mixed precision. Default: ``False``
    warmup_steps
        The number of steps to warmup the learning rate for. Default: 0
    """
    def __init__(self,
                 model: ModelPlugin,
                 config: type[OptConfig],
                 mixed_precision: bool = False,
                 warmup_steps: int = 0) -> None:
        logger.debug(parse_class_init(locals()))
        self._mixed_precision = mixed_precision
        self._accumulation_steps = config.gradient_accumulation()
        self._scaler = None if not mixed_precision else torch.amp.grad_scaler.GradScaler()
        self._clip = None if config.gradient_clipping() == "none" else GradClip(
            T.cast(T.Literal["autoclip", "global_norm", "norm", "value"],
                   config.gradient_clipping()),
            config.clipping_value(),
            config.autoclip_history())

        self._optimizer = self._get_optimizer(model, config)
        self._warmup = None if warmup_steps < 1 else WarmupScheduler(self._optimizer, warmup_steps)
        self._lrf_scheduler: LRFScheduler | None = None

        self.save = T.cast(T.Literal["always", "exit", "never"], config.save_optimizer())
        """`When the optimizer should be saved"""

        self._accumulation_count = 0
        self._session_steps = 0

    @property
    def lrf_scheduler(self) -> LRFScheduler | None:
        """The learning rate scheduler, if learning rate finder is running, otherwise ``None``"""
        return self._lrf_scheduler

    # TODO keep this for weight porting
    @classmethod
    def _get_optimizer_kwargs(cls, config: type[OptConfig]) -> dict[str, T.Any]:
        """Obtain the keyword arguments for the requested optimizer from the user configuration

        Parameters
        ----------
        config
            The optimizer user configuration options

        Returns
        -------
        The optimizer keyword arguments
        """
        retval: dict[str, T.Any] = {"weight_decay": config.weight_decay()}
        name = config.optimizer()

        if name != "lion":
            retval["eps"] = 10 ** config.epsilon_exponent()

        if name in ("adabelief", "adam", "adamw", "adamax", "lion", "nadam"):
            retval["betas"] = (config.ada_beta_1(), config.ada_beta_2())

        if name in ("adabelief", "adam", "adamw"):
            retval["amsgrad"] = config.ada_amsgrad()

        logger.debug("[Optimizer] '%s' kwargs: %s", name, retval)
        return retval

    def _get_optimizer(self, model: ModelPlugin, config: type[OptConfig]) -> torch.optim.Optimizer:
        """Obtain the configured optimizer the given configuration file options

        Parameters
        ----------
        model
            The keras model that is to be trained
        config
            The optimizer user configuration options

        Returns
        -------
        The requested configured optimizer
        """
        name = config.optimizer()
        if name not in _OPTIMIZERS:
            raise ValueError(f"'{name}' is not a valid optimizer. Select from {list(_OPTIMIZERS)}")
        optimizer = _OPTIMIZERS[name]

        retval = optimizer(self._get_parameter_groups(model, config.weight_decay()),
                           lr=config.learning_rate(),
                           **self._get_optimizer_kwargs(config))
        logger.debug("[Optimizer] Got optimizer '%s': %s", name, retval)
        return retval

    def _get_parameter_groups(self, model: ModelPlugin, weight_decay: float
                              ) -> tuple[dict[T.Literal["params", "weight_decay"],
                                              list[nn.Parameter] | float],
                                         dict[T.Literal["params", "weight_decay"],
                                              list[nn.Parameter] | float]]:
        """Obtain the parameter groups from within the keras model

        Parameters
        ----------
        model
            The faceswap model that is to be trained
        weight_decay
            The amount of weight decay to apply

        Returns
        -------
        The parameters that require weight decay in position 0 and no weight decay in position 1
        """
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            dst = no_decay if param.ndim <= 1 or name.endswith(".bias") else decay
            dst.append(param)

        retval: tuple[dict[T.Literal["params", "weight_decay"], list[nn.Parameter] | float],
                      dict[T.Literal["params", "weight_decay"], list[nn.Parameter] | float]] = (
                          {"params": decay, "weight_decay": weight_decay},
                          {"params": no_decay, "weight_decay": 0.0}
                          )

        logger.debug("[Optimizer] decay params: %s, no_decay params: %s",
                     {k: len(v) if isinstance(v, list) else v for k, v in retval[0].items()},
                     {k: len(v) if isinstance(v, list) else v for k, v in retval[1].items()})
        return retval

    def _from_legacy(self,
                     state: dict[str, T.Any]) -> dict[str, T.Any] | None:
        """Populate the remaining param_group items for weights from legacy saved keras optimizer
        and validate shapes

        Parameters
        ----------
        state
            The partial state_dict migrated from a keras optimizer

        Returns
        -------
            The final state_dict grouped for torch or ``None`` if weights could not be mapped
        """
        # TODO move to legacy?
        logger.debug("[Optimizer] Loading weights from legacy Keras optimizer")
        imported_params = state["optimizer"]["state"]
        p_groups = self._optimizer.param_groups
        exists = [p for g in p_groups for p in g["params"]]

        if len(imported_params) != len(exists):
            logger.warning("Imported optimizer weights count mismatch. Optimizer will be reset")
            return None

        for idx, exist in enumerate(exists):
            # exp_avg for ada based optimizers, square_avg for rms-prop
            key = "exp_avg" if "exp_avg" in imported_params[idx] else "square_avg"
            if imported_params[idx][key].shape != exist.shape:
                logger.warning("Imported optimizer weights shape mismatch. "
                               "Optimizer will be reset")
                return None

        imported_p_groups = state["optimizer"]["param_groups"]
        if len(p_groups) != len(imported_p_groups):
            logger.warning("Parameter group count mismatch (exists: %s, imported: %s). "
                           "Optimizer will be reset", len(p_groups), len(imported_p_groups))
            return None

        for idx, group in enumerate(p_groups):
            p_group = state["optimizer"]["param_groups"][idx]
            state["optimizer"]["param_groups"][idx] = {k: p_group.get(k, v)
                                                       for k, v in group.items()}

        return state

    def load_state_dict(self, state_dict: dict[T.Literal["version", "optimizer",
                                                         "scaler", "lrf_scheduler"],
                                               float | dict[str, T.Any]]) -> None:
        """Load the serialized data from a state dict into this object

        Parameters
        ----------
        state_dict
            The serialized data to load
        """
        if not state_dict:
            return
        logger.debug("[Optimizer] Loading state_dict: %s", list(state_dict))

        if state_dict["version"] == 0.5:  # Migrating from keras optimizer
            keras_state = self._from_legacy(state_dict)  # TODO validate
            if keras_state is None:
                return
            state_dict = keras_state

        self._optimizer.load_state_dict(T.cast(dict[str, T.Any], state_dict["optimizer"]))
        if self._scaler is not None and state_dict.get("scaler") is not None:
            logger.debug("[Optimizer] Loading scaler state_dict: %s", state_dict["scaler"])
            self._scaler.load_state_dict(T.cast(dict[str, T.Any], state_dict["scaler"]))

        # Learning Rate Finder resume
        assert self._lrf_scheduler is None, ("LRF_Scheduler should not pre-exist when loading"
                                             "state_dict")
        lrf_dict = T.cast(dict[str, T.Any], state_dict.get("lrf_scheduler"))
        if lrf_dict:
            self._lrf_scheduler = LRFScheduler(self._optimizer,
                                               gamma=1.0,
                                               beta=1.0,
                                               total_steps=1000)
            self._lrf_scheduler.load_state_dict(lrf_dict)
            logger.debug("[Optimizer] Resuming LRF from scheduler: %s",
                         self._lrf_scheduler.state_dict())

    def backward(self, loss: torch.Tensor) -> None:
        """Perform the optimizer's backward pass

        Parameters
        ----------
        loss
            The loss scalar from the forward pass
        """
        scaled = loss / self._accumulation_steps
        if self._scaler:
            self._scaler.scale(scaled).backward()
        else:
            scaled.backward()

    def step(self) -> None:
        """Perform the optimizer step if valid and zero the gradients.

        Handles gradient accumulation, scaling for mixed precision, gradient clipping and the
        learning rate finder

        Parameters
        ----------
        loss
            The total loss scalar from the latest forward pass
        """
        self._accumulation_count += 1
        if self._accumulation_count != self._accumulation_steps:
            return

        if self._clip is not None and self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
        if self._clip is not None:
            self._clip([p for g in self._optimizer.param_groups for p in g["params"]])

        if self._scaler is None:
            self._optimizer.step()
        else:
            self._scaler.step(self._optimizer)
            self._scaler.update()

        if self._lrf_scheduler is not None:
            self._lrf_scheduler.step()
        elif self._warmup is not None and self._session_steps < self._warmup.steps:
            self._session_steps += 1
            self._warmup.step()

        self._optimizer.zero_grad(set_to_none=True)
        self._accumulation_count = 0

    def state_dict(self) -> dict[str, T.Any]:
        """Serialized data as a dict for relevant options contained in this class

        Returns
        -------
        The serialized data for this object for saving and loading
        """
        return {"version": 1.0,
                "optimizer": self._optimizer.state_dict(),
                "scaler": None if self._scaler is None else self._scaler.state_dict(),
                "lrf_scheduler": (None if self._lrf_scheduler is None
                                  else self._lrf_scheduler.state_dict())}

    def to(self, device: torch.Device) -> None:
        """Place the optimizer onto the given device

        Parameters
        ----------
        device
            The device to place the optimizer on to
        """
        logger.debug("[Optimizer] to: %s", device)
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def set_lr(self, lr: float) -> None:
        """Manually assign the optimizer's learning rate with the given value

        Parameters
        ----------
        lr
            The learning rate to apply to the optimizer
        """
        logger.debug("[Optimizer] Setting learning rate to: %s", lr)
        for p in self._optimizer.param_groups:
            p["lr"] = lr
            if "initial_lr" in p:
                p["initial_lr"] = lr

    def enable_learning_rate_finder(self, steps: int, beta: float, start_lr: float, end_lr: float
                                    ) -> LRFScheduler:
        """Enable the Learning Rate Finder on this optimizer to discover the optimal learning rate.
        If a scheduler already exists (from loading state_dict from a resuming LRF session), then
        loaded scheduler is returned. Otherwise a new scheduler is created and returned

        Parameters
        ----------
        steps
            The number of iterations to run the learning rate finder for
        beta
            The amount to smooth accumulated loss by
        start_lr
            The learning rate to start scanning from
        end_lr
            The final learning rate to scan until

        Returns
        -------
        The LearningRate scheduler used for discovering the learning rate
        """
        if self._lrf_scheduler is not None:
            logger.debug("[Optimizer] Resuming saved learning rate scheduler: %s",
                         self._lrf_scheduler)
            return self._lrf_scheduler

        self.set_lr(start_lr)
        gamma: float = (end_lr / start_lr) ** (1.0 / steps)
        self._lrf_scheduler = LRFScheduler(self._optimizer,
                                           gamma=gamma,
                                           beta=beta,
                                           total_steps=steps)
        logger.debug("[Optimizer] Enabled learning rate scheduler: %s", self._lrf_scheduler)
        return self._lrf_scheduler

    def disable_learning_rate_finder(self) -> None:
        """Disables the learning rate finder for the optimizer by deleting the scheduler"""
        del self._lrf_scheduler
        self._lrf_scheduler = None
        logger.debug("[Optimizer] Disabled learning rate scheduler")


__all__ = get_module_objects(__name__)

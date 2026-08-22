#! /usr/bin/env python3
""" Training unit for managing optimizer operations during training

This module contains the core OptimizerUnit class which is responsible for configuring, managing,
and executing optimization processes during model training. It handles various optimizers, gradient
clipping, mixed precision training, and parameter group management
"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model import optimizers
from lib.model.autoclip import AutoClipper
from lib.utils import get_module_objects

from plugins.train import train_config as mod_cfg

from .base import TrainingUnit

if T.TYPE_CHECKING:
    from keras import Variable  # TODO KERAS GOTTA GO
    from torch.optim import Optimizer
    from lib.training.training_loop import TrainStep
    from plugins.train.model.base import ModelPlugin


logger = logging.getLogger(__name__)

_OPTIMIZERS = {"adabelief": optimizers.AdaBelief,
               "adam": torch.optim.Adam,
               "adamax": torch.optim.Adamax,
               "adamw": torch.optim.AdamW,
               "lion": optimizers.Lion,
               "nadam": torch.optim.NAdam,
               "rms-prop": torch.optim.RMSprop}


# TODO keep for legacy weights update
# TODO move to legacy
def get_parameter_group_ids(trainable_variables: list[Variable]
                            ) -> dict[int, T.Literal["decay", "no_decay"]]:
    """ Generate parameter group identifiers for weight decay application from legacy Keras models

    Assigns each trainable variable to either a 'decay' or 'no_decay' group based on its
    dimensionality and name. Bias parameters and 1-dimensional parameters are typically excluded
    from weight decay, while higher dimensional parameters (weights) usually include it

    Parameters
    ----------
    trainable_variables
        List of trainable variables from a Keras model to assign group IDs to

    Returns
    -------
    Dictionary mapping variable indices to their respective group identifiers ('decay' or
    'no_decay')

    Notes
    -----
    This function is primarily kept for legacy weight migration purposes and should not be used in
    new code paths. Keras models used a different grouping scheme than PyTorch optimizers. When
    migrating from Keras to Torch, this maps the old parameter groups to the new
    torch.optim.Optimizer param_groups format
    """
    retval: dict[int, T.Literal["decay", "no_decay"]] = {}
    for idx, var in enumerate(trainable_variables):
        retval[idx] = "no_decay" if var.ndim <= 1 or var.name.endswith("bias") else "decay"

    logger.debug("parameter group ids: %s", retval)
    return retval


class GradClip:
    """ Gradient clipping utility for controlling gradient norms during training

    This class provides various methods for clipping gradients to prevent exploding gradients
    during training. It supports multiple clipping strategies including auto-clipping (adaptive),
    global norm clipping, and value-based clipping with configurable thresholds

    Parameters
    ----------
    method
        The gradient clipping method to use ("autoclip", "global_norm", "norm", or "value")
    value
        The clipping threshold value for the selected method. For autoclip this is the percentile
        to clip at (1.0 = 10th percent, 2.5 = 25th percent). For other methods it's the maximum
        norm/value to apply
    autoclip_history, optional
        History size for auto-clipping (only used when method="autoclip"). Default: 10000
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
        """ Applies gradient clipping based on the L2 norm of parameter gradients

        Parameters
        ----------
        parameters
            List of model parameters whose gradients need clipping
        max_norm
            Maximum allowed norm for gradient vectors
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
        """ Get the appropriate gradient clipping function for the selected method

        Parameters
        ----------
        method : literal["autoclip", "global_norm", "norm", "value"]
            The gradient clipping method to use
        autoclip_history : int
            History size for auto-clipping (only used when method="autoclip")

        Returns
        -------
        A callable that performs gradient clipping based on the selected method

        Raises
        ------
        ValueError
            If an invalid clipper method is specified
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
        """ Apply gradient clipping to a list of model parameters

        Executes the configured gradient clipping operation on the given parameters

        Parameters
        ----------
        parameters
            List of model parameters whose gradients will be clipped
        """
        self._clipper(parameters, self._value)


class OptimizerUnit(TrainingUnit):  # pylint:disable=too-many-instance-attributes
    """ Main optimizer unit that manages training optimization operations

    This unit handles configuration and execution of various optimizers, gradient clipping, mixed
    precision training, parameter grouping, and state management for training processes. It
    interfaces with the training loop to perform weight updates during each iteration

    Parameters
    ----------
    optimizer_name
        Name of the optimizer to use (e.g., "adam", "adamw", "lion", etc.)
    model
        The model plugin containing parameters to optimize
    learning_rate, optional
        Base learning rate for the optimizer. Default: 5e-5
    epsilon_exponent, optional
        Exponent value used for epsilon in optimizers that support it. Default: -7
    mixed_precision, optional
        Whether to use mixed precision training (automatic scaling). Default: ``False``
    accumulation_steps, optional
        Number of gradient accumulation steps before updating parameters. Default: 1
    clipper, optional
        Gradient clipping configuration, if enabled. Default: ``None``
    weight_decay, optional
        L2 Weight decay coefficient for regularization on non-bias parameters. Default: 0.0
    ada_beta_1, optional
        Beta 1 (momentum) parameter for adaptive optimizers (Adam-style). Default: 0.9
    ada_beta_2, optional
        Beta 2 (moving average of squared gradients) parameter for adaptive optimizers (Adam-
        style). Default: 0.999
    ada_amsgrad, optional
        Whether to use AMSGrad variant of adaptive optimizers. Default: ``False``
    """
    def __init__(  # pylint:disable=too-many-arguments,too-many-positional-arguments
            self,
            optimizer_name: str,
            model: ModelPlugin,
            learning_rate: float = 5e-5,
            epsilon_exponent: int = -7,
            mixed_precision: bool = False,
            accumulation_steps: int = 1,
            clipper: GradClip | None = None,
            weight_decay: float = 0.0,
            ada_beta_1: float = 0.9,
            ada_beta_2: float = 0.999,
            ada_amsgrad: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._repr_obj = {"optimizer_name": repr(optimizer_name),
                          "model": repr(model),
                          "learning_rate": repr(learning_rate),
                          "epsilon_exponent": repr(epsilon_exponent),
                          "mixed_precision": repr(mixed_precision),
                          "accumulation_steps": repr(accumulation_steps),
                          "clipper": repr(clipper),
                          "weight_decay": repr(weight_decay),
                          "ada_beta_1": repr(ada_beta_1),
                          "ada_beta_2": repr(ada_beta_2),
                          "ada_amsgrad": repr(ada_amsgrad)}

        self._mixed_precision = mixed_precision
        self._accumulation_steps = accumulation_steps
        self._clip = clipper
        self._scaler = None if not self._mixed_precision else torch.amp.grad_scaler.GradScaler()

        self._optimizer = self._get_optimizer(optimizer_name,
                                              model,
                                              learning_rate=learning_rate,
                                              epsilon_exponent=epsilon_exponent,
                                              weight_decay=weight_decay,
                                              ada_beta_1=ada_beta_1,
                                              ada_beta_2=ada_beta_2,
                                              ada_amsgrad=ada_amsgrad)
        self.save = T.cast(T.Literal["always", "exit", "never"],
                           mod_cfg.Optimizer.save_optimizer())

        self._accumulation_count = 0

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = ", ".join(f"{k}={v}" for k, v in self._repr_obj.items())
        return f"{self.__class__.__name__}({params})"

    @property
    def optimizer(self) -> Optimizer:
        """ The configured PyTorch optimizer instance used for parameter updates """
        return self._optimizer

    # TODO keep this for weight porting
    def _get_optimizer_kwargs(self,
                              name: str,
                              epsilon_exponent: int = -7,
                              weight_decay: float = 0.0,
                              ada_beta_1: float = 0.9,
                              ada_beta_2: float = 0.999,
                              ada_amsgrad: bool = False) -> dict[str, T.Any]:
        """ Generate optimizer-specific keyword arguments for initializing the selected optimizer

        Parameters
        ----------
        name
            Name of the optimizer being configured
        epsilon_exponent, optional
            Exponent value used for epsilon in optimizers that support it. Default: -7
        weight_decay, optional
            Weight decay coefficient for regularization. Default: 0.0
        ada_beta_1, optional
            Beta 1 parameter for adaptive optimizers (Adam-style). Default: 0.9
        ada_beta_2, optional
            Beta 2 parameter for adaptive optimizers (Adam-style). Default: 0.999
        ada_amsgrad, optional
            Whether to use AMSGrad variant of adaptive optimizers. Default: ``False``

        Returns
        -------
        Dictionary of optimizer configuration parameters kwargs
        """
        retval: dict[str, T.Any] = {"weight_decay": weight_decay}

        if name != "lion":
            retval["eps"] = 10 ** epsilon_exponent

        if name in ("adabelief", "adam", "adamw", "adamax", "lion", "nadam"):
            retval["betas"] = (ada_beta_1, ada_beta_2)

        if name in ("adabelief", "adam", "adamw"):
            retval["amsgrad"] = ada_amsgrad

        logger.debug("%s '%s' kwargs: %s", self.log_name, name, retval)
        return retval

    def _get_optimizer(self,
                       name: str,
                       model: ModelPlugin,
                       learning_rate: float = 5e-5,
                       epsilon_exponent: int = -7,
                       weight_decay: float = 0.0,
                       ada_beta_1: float = 0.9,
                       ada_beta_2: float = 0.999,
                       ada_amsgrad: bool = False) -> torch.optim.Optimizer:
        """ Instantiates an optimizer with appropriate parameters for the given model and options

        Parameters
        ----------
        name
            Name of the optimizer to create (e.g., "adam", "adamw")
        model
            The model plugin containing parameters to optimize
        learning_rate, optional
            Base learning rate for the optimizer. Default: 5e-5
        epsilon_exponent, optional
            Exponent value used for epsilon in optimizers that support it. Default: -7
        weight_decay, optional
            Weight decay coefficient for regularization. Default: 0.0
        ada_beta_1, optional
            Beta 1 parameter for adaptive optimizers (Adam-style). Default: 0.9
        ada_beta_2, optional
            Beta 2 parameter for adaptive optimizers (Adam-style). Default: 0.999
        ada_amsgrad, optional
            Whether to use AMSGrad variant of adaptive optimizers. Default: ``False``

        Returns
        -------
        The configured Torch optimizer instance

        Raises
        ------
        ValueError
            If an invalid optimizer name is specified
        """
        if name not in _OPTIMIZERS:
            raise ValueError(f"'{name}' is not a valid optimizer. Select from {list(_OPTIMIZERS)}")
        optimizer = _OPTIMIZERS[name]

        retval = optimizer(self._get_parameter_groups(model, weight_decay),
                           lr=learning_rate,
                           **self._get_optimizer_kwargs(name,
                                                        epsilon_exponent=epsilon_exponent,
                                                        weight_decay=weight_decay,
                                                        ada_beta_1=ada_beta_1,
                                                        ada_beta_2=ada_beta_2,
                                                        ada_amsgrad=ada_amsgrad))
        logger.debug("%s Got optimizer '%s': %s", self.log_name, name, retval)
        return retval

    def _get_parameter_groups(self, model: ModelPlugin, weight_decay: float
                              ) -> tuple[dict[T.Literal["params", "weight_decay"],
                                              list[nn.Parameter] | float],
                                         dict[T.Literal["params", "weight_decay"],
                                              list[nn.Parameter] | float]]:
        """ Divides model parameters into decay and no-decay groups to apply appropriate settings
        (e.g. bias terms typically don't use weight decay).

        Parameters
        ----------
        model
            The model plugin containing parameters to group
        weight_decay
            Weight decay coefficient for parameter groups

        Returns
        -------
        decay
            The parameters that support weight-decay
        no_decay
            The parameters that do not support weight-decay
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

        logger.debug("%s decay params: %s, no_decay params: %s",
                     self.log_name,
                     {k: len(v) if isinstance(v, list) else v for k, v in retval[0].items()},
                     {k: len(v) if isinstance(v, list) else v for k, v in retval[1].items()})
        return retval

    def _from_legacy(self,
                     state: dict[str, T.Any]) -> dict[str, T.Any] | None:
        """ Convert legacy Keras optimizer state to PyTorch format

        Parameters
        ----------
        state
            The legacy state dictionary containing Keras optimizer data

        Returns
        -------
        Converted state dictionary if successful, otherwise ``None`` if migration failed
        """
        # TODO move to legacy?
        logger.debug("%s Loading weights from legacy Keras optimizer", self.log_name)
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

    def on_start(self, loop: TrainStep) -> None:
        """ Initialize optimizer state on the training device

        Moves optimizer internal state tensors to the appropriate device  (CPU/GPU) for training
        execution

        Parameters
        ----------
        loop
            The training step object managing this unit's lifecycle
        """
        logger.debug("%s Moving optimizer to: %s", self.log_name, str(loop.device))
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(loop.device)

    def backward(self, loss: torch.Tensor) -> None:
        """ Perform the optimizer's backward pass

        Computes gradients and applies them to model parameters, handling mixed precision scaling
        and gradient accumulation if enabled

        Parameters
        ----------
        loss
            The computed loss value from the forward pass for backpropagation
        """
        scaled = loss / self._accumulation_steps
        if self._scaler:
            self._scaler.scale(scaled).backward()
        else:
            scaled.backward()

    def step(self, iteration: int) -> None:  # pylint:disable=unused-argument
        """ Execute one optimization step

        Performs:
            - Gradient accumulation scaling
            - Gradient clipping (if configured)
            - Loss scaling/unscaling if mixed precision enabled
            - Updates model parameters
            - Zeros gradients for the next forward pass

        Parameters
        ----------
        iteration
            The current training iteration number
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

        self._optimizer.zero_grad(set_to_none=True)
        self._accumulation_count = 0

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """ Load optimizer state from a saved checkpoint

        Restores the optimizer's internal state including parameter groups, gradient information
        and scaler information from a previously saved state dictionary

        Parameters
        ----------
        state_dict
            The state dictionary containing optimizer configuration and state data

        Notes
        -----
        Handles migration from legacy Keras-based optimizers if needed
        """
        logger.debug("%s Loading state_dict: %s", self.log_name, list(state_dict))

        if state_dict["version"] == 0.5:  # Migrating from keras optimizer
            keras_state = self._from_legacy(state_dict)  # TODO validate
            if keras_state is None:
                return
            state_dict = keras_state

        self._optimizer.load_state_dict(T.cast(dict[str, T.Any], state_dict["optimizer"]))
        if self._scaler is not None and state_dict.get("scaler") is not None:
            logger.debug("%s Loading scaler state_dict: %s", self.log_name, state_dict["scaler"])
            self._scaler.load_state_dict(T.cast(dict[str, T.Any], state_dict["scaler"]))

    def state_dict(self) -> dict[str, T.Any]:
        """ Create a state dictionary for saving optimizer state

        Generates a complete state dictionary containing the optimizer's configuration and internal
        state information, including any scalers that have been applied for mixed precision
        training

        Returns
        -------
        Dictionary containing optimizer state that can be saved to disk
        """
        return {"version": 1.0,
                "optimizer": self._optimizer.state_dict(),
                "scaler": None if self._scaler is None else self._scaler.state_dict()}

    def set_lr(self, lr: float) -> None:
        """ Set a new learning rate for all parameter groups

        Updates the learning rate for all optimizer parameter groups

        Parameters
        ----------
        lr
            The new learning rate value to set
        """
        logger.debug("%s Setting learning rate to: %s", self.log_name, lr)
        for p in self._optimizer.param_groups:
            p["lr"] = lr
            if "initial_lr" in p:
                p["initial_lr"] = lr


__all__ = get_module_objects(__name__)

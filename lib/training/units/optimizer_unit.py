#!/usr/bin/env python3
""" Wraps the selected Torch optimizer and handles optimizer related functions such as loss
scaling, clipping and gradient accumulation """
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model import optimizers
from lib.model.autoclip import AutoClipper
from lib.training.lr_warmup import WarmupScheduler  # TODO
from lib.training.lr_finder import LRFScheduler  # TODO
from lib.utils import get_module_objects

from plugins.train import train_config as mod_cfg

from .base import TrainingUnit

if T.TYPE_CHECKING:
    from keras import Variable
    from lib.training.training_loop import TrainingLoop
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
def get_parameter_group_ids(trainable_variables: list[Variable]
                            ) -> dict[int, T.Literal["decay", "no_decay"]]:
    """ Obtain the index of each item in Keras model's trainable weights that belong to each
    of the optimizer's parameter groups (split by weights that take decay and don't)

    Parameters
    ----------
    trainable_variables
        List of trainable variables from a legacy Keras model

    Returns
    -------
    A dictionary mapping each trainable weight index to its parameter group name:

    - "decay": Regular parameters requiring L2 regularization (non-bias, non-flat tensors)
    - "no_decay": Exempt parameters like biases and flattened layers

    Notes
    -----
    This function is marked for legacy use only. Keras models used a different grouping scheme
    than PyTorch optimizers. When migrating from Keras to Torch, this maps the old parameter
    groups to the new torch.optim.Optimizer param_groups format.

    Examples
    --------
    >>> group_ids = get_parameter_group_ids(model.trainable_variables)
    >>> print(group_ids[0])  # "decay" for most weights
    """
    retval: dict[int, T.Literal["decay", "no_decay"]] = {}
    for idx, var in enumerate(trainable_variables):
        retval[idx] = "no_decay" if var.ndim <= 1 or var.name.endswith("bias") else "decay"

    logger.debug("parameter group ids: %s", retval)
    return retval


class GradClip:
    """ Handles the clipping of gradients based on user supplied parameters

    This class manages different gradient clipping strategies to prevent exploding gradients during
    training. Supports autoclip (adaptive), global norm, per-parameter norm clipping, and value-
    based clipping methods with configurable thresholds.

    Parameters
    ----------
    method
        The clipping method to use: "autoclip", "global_norm", "norm", or "value"
    value
        The clipping threshold. For autoclip this is the percentile to clip at (1.0 = 10th percent,
        2.5 = 25th percent). For other methods it's the maximum norm/value to apply
    autoclip_history
        The history length for auto clipping. Default: 10000

    Notes
    -----
    Autoclip uses exponential moving average of gradient norms to determine appropriate thresholds,
    adapting during training without manual intervention. Other methods use fixed thresholds that
    should be set based on model architecture and loss scale factors.

    Examples
    --------
    >>> clipper = GradClip("global_norm", max_norm=1.0)
    >>> for param in parameters:
    ...     clipper([param])  # Clip all parameters by global norm
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
        """ Clip each parameter independently by its own norm

        Parameters
        ----------
        parameters
            The parameters to clip
        max_norm
            The value to clip by

        Notes
        -----
        This method clips gradients on a per-parameter basis rather than globally, which can be
        useful when different layers benefit from different clipping thresholds.
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
        """ Obtain the correct function to clip the gradients based on the selected method

        Parameters
        ----------
        method
            The clipping method to use
        autoclip_history
            The history length for auto clipping

        Returns
        -------
        The function used to clip the gradients

        Notes
        -----
        Maps each method string to its corresponding clipping implementation. Autoclip uses a
        custom AutoClipper that tracks gradient norms over time, while other methods use PyTorch's
        built-in utilities or our per-parameter norm clipping implementation.

        Raises
        ------
        ValueError
            If an unrecognized method is provided
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
        """ Clip the given parameters by the chosen method

        Parameters
        ----------
        parameters
            The parameters to clip

        Notes
        -----
        This is a callable wrapper that accepts parameters as a single argument. It uses
        the `clipping_value` as the clipping threshold passed to the underlying clipper function.

        Examples
        --------
        >>> grad_clipper = GradClip("global_norm", 1.0)
        >>> grad_clipper(model.parameters())  # Clip all model gradients
        """
        self._clipper(parameters, self._value)


class OptimizerUnit(TrainingUnit):  # pylint:disable=too-many-instance-attributes
    """ Object for managing the selected Torch optimizer

    This unit wraps PyTorch optimizers with additional functionality: gradient accumulation,
    mixed precision scaling (AMP), learning rate scheduling, warmup phases, and gradient clipping.

    Parameters
    ----------
    optimizer_name
        The name of the optimizer to use ("adabelief", "adam", "adamw", "adamax", "lion", etc.)
    model
        The Torch model that is to be trained
    learning_rate
        The base learning rate. Default: 5e-5
    epsilon_exponent
        Log-space epsilon for Adam family optimizers. Default: -7
    mixed_precision
        ``True`` to use automatic mixed precision training with GradScaler. Default: ``False``
    warmup_steps
        Number of steps to linearly warm up learning rate from zero. Default: 0
    accumulation_steps
        Gradient accumulation factor (processes N batches before optimizer step). Default: 1
    clipper
        Optional gradient clipping instance. Default: ``None``
    weight_decay
        L2 regularization coefficient applied to non-bias parameters. Default: 0.0
    ada_beta_1
        Beta1 parameter for Adam family optimizers (momentum). Default: 0.9
    ada_beta_2
        Beta2 parameter for Adam family optimizers (moving average of squared gradients).
        Default: 0.999
    ada_amsgrad
        Whether to use AMSGrad variant for better stability with adaptive moments.
        Default: ``False``

    Notes
    -----
    This unit handles the complete optimization lifecycle including:

    - Gradient accumulation across multiple batches before stepping
    - Learning rate warmup during initial training phase
    - Automatic mixed precision scaling when enabled
    - Gradient clipping to prevent exploding gradients
    - Learning Rate Finder integration for optimal LR discovery

    Examples
    --------
    >>> opt_unit = OptimizerUnit("adamw", model, learning_rate=1e-4)
    >>> opt_unit.on_start(training_loop)  # Move optimizer to device
    >>> loss.backward()
    >>> opt_unit.step(iteration=100)  # Perform optimization step
    """
    def __init__(  # pylint:disable=too-many-arguments,too-many-positional-arguments
            self,
            optimizer_name: str,
            model: ModelPlugin,
            learning_rate: float = 5e-5,
            epsilon_exponent: int = -7,
            mixed_precision: bool = False,
            warmup_steps: int = 0,
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
                          "warmup_steps": repr(warmup_steps),
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
        self._warmup = None if warmup_steps < 1 else WarmupScheduler(self._optimizer, warmup_steps)
        self._lrf_scheduler: LRFScheduler | None = None

        self.save = T.cast(T.Literal["always", "exit", "never"],
                           mod_cfg.Optimizer.save_optimizer())
        """ When the optimizer should be saved """

        self._accumulation_count = 0
        self._session_steps = 0

    def __repr__(self) -> str:
        """ String representation for debugging and logging """
        params = ", ".join(f"{k}={v}" for k, v in self._repr_obj.items())
        return f"{self.__class__.__name__}({params})"

    @property
    def lrf_scheduler(self) -> LRFScheduler | None:
        """ The learning rate scheduler, if learning rate finder is running, otherwise ``None`` """
        return self._lrf_scheduler

    # TODO keep this for weight porting
    def _get_optimizer_kwargs(self,
                              name: str,
                              epsilon_exponent: int = -7,
                              weight_decay: float = 0.0,
                              ada_beta_1: float = 0.9,
                              ada_beta_2: float = 0.999,
                              ada_amsgrad: bool = False) -> dict[str, T.Any]:
        """ Obtain the keyword arguments for the requested optimizer from user configuration

        Parameters
        ----------
        name
            The optimizer class name (not used directly but passed to constructor)
        epsilon_exponent
            Log-space epsilon value. Default: -7
        weight_decay
            L2 regularization coefficient. Default: 0.0
        ada_beta_1
            Beta1 for Adam family optimizers. Default: 0.9
        ada_beta_2
            Beta2 for Adam family optimizers. Default: 0.999
        ada_amsgrad
            Whether to use AMSGrad variant. Default: ``False``

        Returns
        -------
        A dictionary of keyword arguments compatible with PyTorch optimizer constructors

        Notes
        -----
        Different optimizers require different kwargs:

        - Adam family (adam, adamw, adabelief): eps, betas, amsgrad
        - RMSprop: eps only
        - Adamax: eps and betas
        - Lion/NAdam: no special kwargs beyond standard
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
        """ Create and configure the optimizer based on user settings

        Parameters
        ----------
        name
            The name of the optimizer to load ("adam", "adamw", etc.)
        model
            The Torch model that is to be trained
        learning_rate
            The base learning rate. Default: 5e-5
        epsilon_exponent
            Log-space epsilon for Adam family optimizers. Default: -7
        weight_decay
            L2 regularization coefficient applied to non-bias parameters. Default: 0.0
        ada_beta_1
            Beta1 parameter for Adam family optimizers. Default: 0.9
        ada_beta_2
            Beta2 parameter for Adam family optimizers. Default: 0.999
        ada_amsgrad
            Whether to use AMSGrad variant. Default: ``False``

        Returns
        -------
        The configured PyTorch optimizer instance ready for training

        Notes
        -----
        Separates parameters into decay and no_decay groups based on:

        - Non-bias tensors (requires weight decay)
        - Bias terms and flattened layers (exempt from weight decay)
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
        """ Separate model parameters into decay and no_decay groups for optimizer

        Parameters
        ----------
        model
            The Torch model that is to be trained
        weight_decay
            L2 regularization coefficient applied to non-bias parameters

        Returns
        -------
        A tuple of two parameter group dictionaries:
        - Position 0: Parameters with weight decay (non-flat, non-bias)
        - Position 1: Parameters without weight decay (bias, flattened layers)

        Notes
        -----
        PyTorch optimizers expect parameters organized by regularization type. This method
        automatically groups them based on dimensionality and parameter names to ensure proper
        L2 regularization application.
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
        """ Populate remaining parameter groups from legacy Keras optimizer weights

        Parameters
        ----------
        state
            The partial state_dict migrated from a Keras optimizer

        Returns
        -------
        The final state_dict grouped for torch or ``None`` if weights could not be mapped

        Notes
        -----
        This method handles the migration of legacy Keras optimizer states to PyTorch format. It
        validates parameter counts and shapes before applying weights, returning None if any check
        fails and warning users that training will restart with new initializations.
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

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """ Load the serialized data from a state dict into this object

        Parameters
        ----------
        state_dict
            The serialized data to load (typically from a checkpoint file)

        Notes
        -----
        Handles version 0.5 legacy Keras optimizer states by migrating them first, then loads:

        - Optimizer weights and momentum buffers
        - AMP GradScaler state when mixed precision is enabled
        - Learning Rate Finder scheduler if resuming interrupted LRF session

        Examples
        --------
        >>> opt_unit.load_state_dict(torch.load("checkpoint.pth"))  # Resumes from saved state
        """
        if not state_dict:
            return
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
            logger.debug("%s Resuming LRF from scheduler: %s",
                         self.log_name, self._lrf_scheduler.state_dict())

    def on_start(self, loop: TrainingLoop) -> None:
        """ Move the optimizer to the training device

        Parameters
        ----------
        loop
            The active TrainingLoop instance. Used to access the shared device context.

        Notes
        -----
        This method transfers all tensor state variables (momentum buffers, velocity terms, etc.)
        from CPU to GPU when using CUDA or other accelerators. Essential for distributed training
        and mixed precision setups where tensors must reside on the same device as parameters.
        """
        logger.debug("%s Moving optimizer to: %s", self.log_name, str(loop.device))
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(loop.device)

    def backward(self, loss: torch.Tensor) -> None:
        """ Perform the optimizer's backward pass

        Parameters
        ----------
        loss
            The loss scalar from the forward pass

        Notes
        -----
        Applies gradient accumulation before computing gradients. When using mixed precision (AMP),
        scales the loss appropriately to maintain numerical stability across training steps.

        Notes
        -----
        Called once per batch before optimizer.step(). With gradient accumulation, this is called
        multiple times before an actual optimization step occurs, with the loss scaled accordingly
        """
        scaled = loss / self._accumulation_steps
        if self._scaler:
            self._scaler.scale(scaled).backward()
        else:
            scaled.backward()

    def step(self, iteration: int) -> None:
        """ Perform the optimizer step if valid and zero the gradients

        Parameters
        ----------
        iteration
            The current total iteration count

        Notes
        -----
        This method orchestrates the complete optimization cycle including:

        1. Gradient accumulation check (only steps every N batches)
        2. Unscales gradients when using AMP for accurate updates
        3. Applies gradient clipping if configured (prevents exploding gradients)
        4. Performs optimizer step with optional scaler update
        5. Advances LR finder scheduler or warmup phase as appropriate
        6. Zeros gradients in preparation for next iteration

        Notes
        -----
        Learning rate finder runs (iteration < 0) skip accumulation and step logic, focusing only
        on collecting loss values across different learning rates.
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
        """ Serialize the optimizer and related state for saving to checkpoint

        Returns
        -------
        A dictionary containing:

        - version: The serialization format version (1.0)
        - optimizer: Complete optimizer state including momentum buffers
        - scaler: AMP GradScaler state when mixed precision is enabled
        - lrf_scheduler: Learning Rate Finder scheduler if active

        Notes
        -----
        This method captures all trainable state needed to resume training from a checkpoint,
        including gradient accumulators and learning rate schedules.

        Examples
        --------
        >>> checkpoint = optimizer.state_dict()  # Save optimizer state with model weights
        """
        return {"version": 1.0,
                "optimizer": self._optimizer.state_dict(),
                "scaler": None if self._scaler is None else self._scaler.state_dict(),
                "lrf_scheduler": (None if self._lrf_scheduler is None
                                  else self._lrf_scheduler.state_dict())}

    def set_lr(self, lr: float) -> None:
        """ Manually assign the optimizer's learning rate with the given value

        Parameters
        ----------
        lr
            The learning rate to apply to all parameter groups

        Notes
        -----
        Updates both current and initial learning rates if they exist.

        Examples
        --------
        >>> optimizer.set_lr(1e-3)  # Set to 0.001 for faster convergence phase
        """
        logger.debug("%s Setting learning rate to: %s", self.log_name, lr)
        for p in self._optimizer.param_groups:
            p["lr"] = lr
            if "initial_lr" in p:
                p["initial_lr"] = lr

    def enable_learning_rate_finder(self, steps: int, beta: float, start_lr: float, end_lr: float
                                    ) -> LRFScheduler:
        """ Enable the Learning Rate Finder on this optimizer to discover optimal learning rate

        Parameters
        ----------
        steps
            The number of iterations to run the learning rate finder for
        beta
            The amount to smooth accumulated loss by (exponential moving average)
        start_lr
            The learning rate to start scanning from
        end_lr
            The final learning rate to scan until

        Returns
        -------
        The LearningRate scheduler used for discovering the optimal learning rate

        Notes
        -----
        If a scheduler already exists (from loading state_dict during resume), returns existing
        instance instead of creating new one. Otherwise initializes fresh scheduler with
        exponential decay from start_lr to end_lr over specified steps.

        Examples
        --------
        >>> lr_scheduler = self.enable_learning_rate_finder(100, 0.9, 1e-6, 1e-2)

        Notes
        -----
        After enabling LRF, use on_save() at each iteration to record loss values for finding
        optimal learning rate.  # TODO is this nonsense?
        """
        if self._lrf_scheduler is not None:
            logger.debug("%s Resuming saved learning rate scheduler: %s",
                         self.log_name, self._lrf_scheduler)
            return self._lrf_scheduler

        self.set_lr(start_lr)
        gamma: float = (end_lr / start_lr) ** (1.0 / steps)
        self._lrf_scheduler = LRFScheduler(self._optimizer,
                                           gamma=gamma,
                                           beta=beta,
                                           total_steps=steps)
        logger.debug("%s Enabled learning rate scheduler: %s", self.log_name, self._lrf_scheduler)
        return self._lrf_scheduler

    def disable_learning_rate_finder(self) -> None:
        """ Disable the Learning Rate Finder for this optimizer

        Notes
        -----
        Removes the LR finder scheduler and prepares optimizer for normal training. Call
        after completing LRF scan to resume regular gradient descent with discovered optimal
        learning rate set via state_dict or handle_lr_finder_completion().
        """
        del self._lrf_scheduler
        self._lrf_scheduler = None
        logger.debug("%s Disabled learning rate scheduler", self.log_name)


__all__ = get_module_objects(__name__)

#! /usr/env/bin/python3
"""Run the training loop for a training plugin"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from threading import Event, Lock
import typing as T

import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from lib.multithreading import FSThread
from lib.torch_utils import get_device
from plugins.train import train_config as mod_cfg
from plugins.train.trainer.base import TrainerPlugin

from .data import TrainLoader
from .units import EventUnit, LoadUnit, LossUnit, PluginUnit, SaveUnit, SnapshotUnit, StateUnit
from .units.core.optimizer_unit import GradClip, OptimizerUnit
# from .lr_finder import LearningRateFinder
from .units import TrainingUnit

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.model.plugin.handler import FaceswapModel
    from plugins.train.model.base import ModelPlugin
    from .loss import BatchLoss

logger = logging.getLogger(__name__)


# TODO ping-pong

@dataclass
class TrainerReturn:
    """Return object from training loop to calling script"""
    exit: bool = False
    """``True`` to exist training"""
    preview_image: npt.NDArray[np.uint8] | None = None
    """Generated preview image if one should be shown"""
    preview_title: str = ""
    """Title for the generated preview image if one should be shown"""


UnitGroupT = T.Literal["core", "optional"]
UnitStageT = T.Literal["start", "step", "save", "update", "end"]
UnitStageDictT = dict[UnitStageT, list[TrainingUnit]]


@dataclass
class Units:
    """Container for organizing training units across lifecycle stages

    This class provides a structured way to group and manage all training loop units by their
    stage of execution (start, step, save, end) and category (core, optional). It automatically
    aggregates all units into an accessible dictionary structure for easy iteration.

    Attributes
    ----------
    stages_core
        Core units that are essential to the training loop at each lifecycle stage. These run
        regardless of user configuration changes

    stages_optional
        Optional units that can be enabled/disabled through configuration (e.g., preview,
        timelapse, external callbacks)

    Notes
    -----
    Units are organized into two categories and five lifecycle stages:

    Categories:

    - **core**: Essential training components that always execute (loss tracking, saves, optimizer)
    - **optional**: Optional features like GUI preview or timelapse generation

    Lifecycle Stages (executed in order):

    1. **start**: Setup and configuration before first batch
    2. **step**: Per-iteration processing after backpropagation
    3. **save**: Actions taken when save intervals are reached
    4. **update**: Actions taken after a save has completed or on user intervention
    5. **end**: Cleanup operations on training completion

    Examples
    --------
    >>> units = Units()
    >>> for unit in units.on_start:  # Iterate start-stage units
    ...     unit.on_start(training_loop)
    """
    stages_core: UnitStageDictT = field(
        default_factory=lambda: T.cast(
            UnitStageDictT, {"start": [], "step": [], "save": [], "update": [], "end": []}
            )
        )
    """ Core units that are essential to the training loop at each lifecycle stage """

    stages_optional: UnitStageDictT = field(
        default_factory=lambda: T.cast(
            UnitStageDictT, {"start": [], "step": [], "save": [], "update": [], "end": []}
            )
        )
    """ Optional units that can be enabled/disabled through configuration """

    _all: dict[UnitGroupT, dict[str, TrainingUnit]] | None = field(init=False, default=None)

    @property
    def all(self) -> dict[UnitGroupT, dict[str, TrainingUnit]]:
        """ All units organized by category (core/optional) as nested dictionaries:

        {
            "core": {"PluginUnit": <unit>, "LossUnit": <unit>, ...},
            "optional": {"PreviewUnit": <unit> if enabled, ...}
        }
        Units are arbitrarily ordered within each category
        """
        if self._all is None:
            self._all = {}
            for key in T.cast(list[UnitGroupT], ["core", "optional"]):
                units = set(y for x in getattr(self, f"stages_{key}").values() for y in x)
                self._all[key] = {u.__class__.__name__: u for u in units}
            logger.debug("[Units] All units: %s", self._all)
        return self._all

    @property
    def core(self) -> dict[str, TrainingUnit]:
        """ Dictionary mapping unit class names to their instances """
        return self.all["core"]

    @property
    def optional(self) -> dict[str, TrainingUnit]:
        """ Dictionary mapping unit class names to their instances """
        return self.all["optional"]

    @property
    def on_start(self) -> list[TrainingUnit]:
        """ Combined list of core and optional units configured to execute before the first batch.
        Core units are ordered first, then optional units in the order they were provided """
        load_unit = next(x for x in self.stages_core["start"] if isinstance(x, LoadUnit))
        core = [x for x in self.stages_core["start"] if x != load_unit]
        return core + self.stages_optional["start"] + [load_unit]

    @property
    def step(self) -> list[TrainingUnit]:
        """ Combined list of core and optional units configured to execute after every
        backpropagation. Core units are ordered first, then optional units in the order they were
        provided """
        return self.stages_core["step"] + self.stages_optional["step"]

    @property
    def on_save(self) -> list[TrainingUnit]:
        """ Combined list of optional and core save-stage units. Optional units are ordered first
        in the order they were provided followed by core units """
        return self.stages_optional["save"] + self.stages_core["save"]

    @property
    def on_update(self) -> list[TrainingUnit]:
        """ Combined list of optional and core update-stage units. Optional units are ordered first
        in the order they were provided followed by core units """
        return self.stages_optional["update"] + self.stages_core["update"]

    @property
    def on_end(self) -> list[TrainingUnit]:
        """ Combined list of optional and end-stage core units for cleanup operations. Optional
        units are ordered first in the order they were provided followed by core units"""
        return self.stages_optional["end"] + self.stages_core["end"]

    @property
    def have_state_dict(self) -> dict[str, TrainingUnit]:
        return {k: v for k, v in self.core.items() | self.optional.items()
                if v.has_state_dict}

    def add_unit(self, unit: TrainingUnit, is_core: bool) -> None:
        """Register a training unit to its appropriate lifecycle stages

        Parameters
        ----------
        unit
            The training unit to register. This should be a ``TrainingUnit`` subclass that declares
            its capabilities via overriding the stage methods (on_start, step, on_save, or on_end)

        is_core
            ``True`` if the unit belongs to core functionality or ``False`` for optional

        Notes
        -----
        Training units are organized into four lifecycle stages:

        - **start**: Units configured when training begins (typically for setup)
        - **step**: Units executed on each training iteration step
        - **save**: Units that are executed at each save interval
        - **update**: Units that are executed after a save has completed or on user intervention
        - **end**: Units that perform cleanup operations when training ends

        The method checks which stages the unit supports. Units are appended to
        ``self._units[stage]`` lists in order. For `on_start` and `step` core units are executed
        first. For `on_save` and `on_end` core units are executed last. Optional units are executed
        in the order they are added to the loop.

        Examples
        --------
        >>> custom_unit = CustomLossUnit()
        >>> self.units.add_unit(custom_unit, is_core=False)
        """
        stage_group = self.stages_core if is_core else self.stages_optional
        for key in ("start", "step", "save", "update", "end"):
            if not getattr(unit, f"has_{key}"):
                continue
            logger.debug("[Units] '%s' Adding 'stage_%s'['%s']",
                         unit.__class__.__name__, "core" if is_core else "optional", key)
            stage_group[key].append(unit)


class TrainStep:  # pylint:disable=too-many-instance-attributes
    """ Handles the feeding of training images to Faceswap models, collating the loss and running
    any training step units

    Parameters
    ----------
    faceswap_model
        The object that holds the Faceswap Torch nn.module, its state and its io operations
    trainer
        The object responsible for forward and backwards passes through the model
    loader
        The data loader for feeding training data to the model
    training_events
        The event triggers for communicating with the main thread
    save_interval
        The number of steps between each model save. Default: 250
    snapshot_interval
        The number of steps between full model checkpoint snapshots. Default: 25000
    """
    def __init__(self,
                 faceswap_model: FaceswapModel,
                 trainer: TrainerPlugin,
                 loader: TrainLoader,
                 training_events: TrainingEvents,
                 save_interval: int = 250,
                 snapshot_interval: int = 25000) -> None:
        logger.debug(parse_class_init(locals()))

        self._model = faceswap_model
        self._events = training_events

        self._started = False
        self._device = get_device()
        self._units = Units()

        trainer.set_training_precision(mod_cfg.mixed_precision(), str(self._device))
        self._optimizer_unit = self._get_optimizer(faceswap_model.plugin)
        self._create_base_units(loader=loader,
                                trainer=trainer,
                                events=training_events,
                                save_interval=save_interval,
                                snapshot_interval=snapshot_interval)

    @property
    def device(self) -> torch.Device:
        """ The device that the model is training on """
        return self._device

    @property
    def events(self) -> TrainingEvents:
        """ The event signaller for internal and external triggers """
        return self._events

    @property
    def model(self) -> FaceswapModel:
        """ The object that manages the FaceswapModel plugin and state """
        return self._model

    @property
    def units(self) -> Units:
        """ The life-cycle units that are being executed """
        return self._units

    @property
    def optimizer_unit(self) -> OptimizerUnit:
        """ The Optimizer being used for training """
        return self._optimizer_unit

    @property
    def iteration(self) -> int:
        """ The current total training step """
        return T.cast(StateUnit, self._units.core["StateUnit"]).iteration

    @property
    def current_loss(self) -> list[BatchLoss]:
        """ A list of BatchLoss objects containing the detached loss outputs for each identity
        processed during this iteration. The list persists, so a reference to this object can be
        safely taken and it will always contain the loss for the current step.

        Notes
        -----
        Values are populated after trainer.step() is called and cleared at the start of each
        new step to ensure accurate per-iteration tracking across the training session."""
        return T.cast(PluginUnit, self._units.core["PluginUnit"]).current_loss

    def _get_optimizer(self, model: ModelPlugin) -> OptimizerUnit:
        clipping = mod_cfg.Optimizer.gradient_clipping()
        assert clipping in ("autoclip", "global_norm", "norm", "value", "none")
        clipper = None if clipping == "none" else GradClip(clipping,
                                                           mod_cfg.Optimizer.clipping_value(),
                                                           mod_cfg.Optimizer.autoclip_history())
        retval = OptimizerUnit(
            optimizer_name=mod_cfg.Optimizer.optimizer(),
            model=model,
            learning_rate=mod_cfg.Optimizer.learning_rate(),
            epsilon_exponent=mod_cfg.Optimizer.epsilon_exponent(),
            mixed_precision=mod_cfg.mixed_precision(),
            accumulation_steps=mod_cfg.Optimizer.gradient_accumulation(),
            clipper=clipper,
            weight_decay=mod_cfg.Optimizer.weight_decay(),
            ada_beta_1=mod_cfg.Optimizer.ada_beta_1(),
            ada_beta_2=mod_cfg.Optimizer.ada_beta_2(),
            ada_amsgrad=mod_cfg.Optimizer.ada_amsgrad()
            )
        logger.debug("[TrainStep] Built optimizer: %s", retval)
        return retval

    def _get_plugin_unit(self,
                         loader: TrainLoader,
                         trainer: TrainerPlugin,
                         optimizer: OptimizerUnit) -> PluginUnit:
        """ Configure and create the PluginUnit that runs the forwards and backwards pass through
        the model

        Parameters
        ----------
        loader
            The data loader for feeding training data to the model
        trainer
            The object responsible for forward and backwards passes through the model
        optimizer
            The optimizer to use for training

        Returns
        -------
        The configured PluginUnit for the training loop
        """
        loss_funcs = [mod_cfg.Loss.loss_function(),
                      mod_cfg.Loss.loss_function_2(),
                      mod_cfg.Loss.loss_function_3(),
                      mod_cfg.Loss.loss_function_4()]
        loss_weights = [x / 100. for x in (100,
                                           mod_cfg.Loss.loss_weight_2(),
                                           mod_cfg.Loss.loss_weight_3(),
                                           mod_cfg.Loss.loss_weight_4())]
        retval = PluginUnit(loader=loader,
                            trainer=trainer,
                            optimizer=optimizer,
                            model=self._model,
                            device=self._device,
                            loss_functions=dict(zip(loss_funcs, loss_weights)),
                            penalize_mask_loss=mod_cfg.Loss.penalized_mask_loss(),
                            eye_multiplier=mod_cfg.Loss.eye_multiplier(),
                            mouth_multiplier=mod_cfg.Loss.mouth_multiplier(),
                            mask_loss=(None if not mod_cfg.Loss.learn_mask()
                                       else mod_cfg.Loss.mask_loss_function()))
        logger.debug("[TrainStep] Built PluginUnit: %s", retval)
        return retval

    def _create_base_units(self,
                           loader: TrainLoader,
                           trainer: TrainerPlugin,
                           events: TrainingEvents,
                           save_interval: int,
                           snapshot_interval: int) -> None:
        """ Create and register default training units for loss tracking, snapshots, and saves

        Parameters
        ----------
        loader
            The data loader for feeding training data to the model
        trainer
            The object responsible for forward and backwards passes through the model
        save_interval
            The number of iterations between each model save
        snapshot_interval
            The number of steps between full model checkpoint snapshots. Only creates a
            ``SnapshotUnit`` if this interval is positive.

        Notes
        -----
        This method instantiates the three core saving units:

        1. **LossUnit** - Always created to track loss values and detect convergence
        2. **SnapshotUnit** - Created only when ``snapshot_interval > 0`` for periodic
           checkpointing (e.g., every 25,000 iterations)
        3. **SaveUnit** - Always added last as the final save operation that handles both
           loss-based backups and exit checkpoints

        All units are registered with their respective lifecycle stages ("start", "step",
        "save", or "end") through the ``add_unit`` method based on their capabilities.

        The order of unit registration ensures proper coordination during training:
        LossUnit runs at each step, SnapshotUnit at intervals, and SaveUnit always runs
        last to finalize any pending saves.
        """
        save_optimizer = T.cast(T.Literal["always", "never", "exit"],
                                mod_cfg.Optimizer.save_optimizer())

        plugin_unit = self._get_plugin_unit(loader, trainer, self.optimizer_unit)
        loss_unit = LossUnit(mod_cfg.nan_protection(), plugin_unit.current_loss, self._device)
        save_unit = SaveUnit(self.model,
                             self.optimizer_unit,
                             events,
                             loss_unit.current_average,
                             save_interval,
                             save_optimizer)

        self._units.add_unit(StateUnit(self._model.state, trainer.batch_size), is_core=True)
        self._units.add_unit(plugin_unit, is_core=True)
        self._units.add_unit(self.optimizer_unit, is_core=True)
        self._units.add_unit(loss_unit, is_core=True)
        self._units.add_unit(EventUnit(events, save_interval), is_core=True)

        if snapshot_interval > 0:
            self._units.add_unit(SnapshotUnit(self._model, self.optimizer_unit, snapshot_interval),
                                 is_core=True)

        self._units.add_unit(LoadUnit(self.model), is_core=True)
        self._units.add_unit(save_unit, is_core=True)

    def add_unit(self, unit: TrainingUnit) -> None:
        """ Register a training unit to its appropriate lifecycle stage

        Parameters
        ----------
        unit
            The training unit to register. This should be a ``TrainingUnit`` subclass that declares
            its capabilities via overriding the stage methods ``on_start``, ``step``, ``on_save``,
            or ``on_end``)

        Notes
        -----
        Training units are organized into four lifecycle stages:

        - **start**: Units configured when training begins (typically for setup)
        - **step**: Units executed on each training iteration step
        - **save**: Units that are executed at each save interval
        - **update**: Units that are executed after a save has completed or on user intervention
        - **end**: Units that perform cleanup operations when training ends

        The method checks which stages the unit supports. Units are appended to
        ``self._units[stage]`` lists in order. Outside of LossUnit (which is always executed first)
        and SaveUnit (which is always executed last), units are executed in the order they are
        added to the loop
        """
        self._units.add_unit(unit, is_core=False)

    def _on_train_start(self) -> None:
        """ Start the training loop by executing all on_start units """
        logger.debug("[TrainStep] Starting")
        for unit in self._units.on_start:
            logger.debug("[TrainStep] Executing on_start: '%s'", unit.__class__.__name__)
            unit.on_start(self)

        self._started = True

    def _step(self) -> None:
        for unit in self._units.step:
            logger.trace("[TrainStep] %s step %s",  # type:ignore[attr-defined]
                         unit.__class__.__name__, self.iteration)
            unit.step(self.iteration)

    def _save(self) -> None:
        for unit in self._units.on_save:
            logger.debug("[TrainStep] %s Saving step %s", unit.__class__.__name__, self.iteration)
            unit.on_save(self.iteration)
        self._events.save.clear()

    def _update(self) -> None:
        for unit in self._units.on_update:
            logger.debug("[TrainStep] %s Updating", unit.__class__.__name__)
            unit.on_update()
        self._events.update.clear()

    def step(self) -> None:
        if not self._started:
            # TODO We currently have a problem here if both lrf + warmup selected. Warmup sets the
            # initial LR to 0.0 even if step() not called
            self._on_train_start()

        logger.trace("[TrainStep] step %s",  self.iteration)  # type:ignore[attr-defined]
        self._step()

        if self._events.save.is_set():
            self._save()

        if self._events.update.is_set():
            self._update()

    def on_end(self) -> None:
        for unit in self._units.on_end:
            logger.trace("[TrainStep] ending %s",  # type:ignore[attr-defined]
                         unit.__class__.__name__)
            unit.on_end()
        logger.debug("[TrainStep] Training ended")


@dataclass
class TrainingEvents:
    save: Event = field(default_factory=Event)
    exit: Event = field(default_factory=Event)
    update: Event = field(default_factory=Event)
    toggle_mask: Event = field(default_factory=Event)
    _preview: None | npt.NDArray[np.uint8] = None
    _lock: Lock = field(default_factory=Lock)

    def get_preview(self) -> None | npt.NDArray[np.uint8]:
        with self._lock:
            if self._preview is None:
                return None
            retval = self._preview
            self._preview = None
        logger.debug("[TrainingEvents] Getting preview: %s", retval.shape)
        return retval

    def set_preview(self, preview: npt.NDArray[np.uint8]) -> None:
        logger.debug("[TrainingEvents] Setting preview: %s", preview.shape)
        with self._lock:
            self._preview = preview


class TrainingLoop:
    def __init__(self,
                 iterations: int,
                 faceswap_model: FaceswapModel,
                 trainer: TrainerPlugin,
                 loader: TrainLoader,
                 save_interval: int = 250,
                 snapshot_interval: int = 25000) -> None:
        logger.debug(parse_class_init(locals()))
        self._iterations = iterations
        self._events = TrainingEvents()
        self._stepper = TrainStep(faceswap_model,
                                  trainer=trainer,
                                  loader=loader,
                                  training_events=self._events,
                                  save_interval=save_interval,
                                  snapshot_interval=snapshot_interval)
        self._thread = FSThread(self._main_loop, "training_loop")

    @property
    def events(self) -> TrainingEvents:
        return self._events

    def check_and_re_raise_error(self) -> None:
        self._thread.check_and_raise_error()

    def _training_loop(self) -> None:
        for _ in range(self._iterations):  # TODO need to track iters because of LRF
            if self._events.exit.is_set():
                logger.debug("[TrainingLoop] Exit requested")
                break

            self._stepper.step()

        self._stepper.on_end()
        logger.debug("[TrainingLoop] Training Complete")

    def _main_loop(self) -> None:
        """Main loop of the training thread"""
        logger.debug("[TrainingLoop] Commencing Training")
        try:
            self._training_loop()
        except KeyboardInterrupt:
            try:
                logger.info("[Train] Keyboard Interrupt Caught. Saving Weights and exiting")
                self._stepper.on_end()
            except KeyboardInterrupt:
                logger.warning("Saving model weights has been cancelled!")
        except Exception as err:
            raise err

    def add_unit(self, unit: TrainingUnit) -> None:
        """ Register a training unit to its appropriate lifecycle stage

        Parameters
        ----------
        unit
            The training unit to register. This should be a ``TrainingUnit`` subclass that declares
            its capabilities via overriding the stage methods ``on_start``, ``step``, ``on_save``,
            or ``on_end``)

        Notes
        -----
        Training units are organized into four lifecycle stages:

        - **start**: Units configured when training begins (typically for setup)
        - **step**: Units executed on each training iteration step
        - **save**: Units that are executed at each save interval
        - **update**: Units that are executed after a save has completed or on user intervention
        - **end**: Units that perform cleanup operations when training ends

        The method checks which stages the unit supports. Units are appended to
        ``self._units[stage]`` lists in order. Outside of LossUnit (which is always executed first)
        and SaveUnit (which is always executed last), units are executed in the order they are
        added to the loop
        """
        self._stepper.add_unit(unit)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()


__all__ = get_module_objects(__name__)

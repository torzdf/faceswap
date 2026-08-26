#! /usr/env/bin/python3
"""TrainingLoop - Core training loop orchestrator that executes model training

This module contains the main training infrastructure including TrainingLoop which runs
training in a separate thread, TrainStep which manages individual iterations and unit lifecycle,
and Units which organize all training units by their activation stage. The system uses an event-
driven design where TrainingEvents enables communication between the training thread (background)
and main thread (foreground)
"""
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
from .units import EventUnit, LoadUnit, LossUnit, PluginUnit, SaveUnit, StateUnit
from .units.core.optimizer_unit import GradClip, OptimizerUnit
from .units import TrainingUnit

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.model.plugin import FaceswapModel
    from plugins.train.model.base import ModelPlugin
    from .loss import BatchLoss

logger = logging.getLogger(__name__)


# TODO ping-pong
# TODO rejig units + preview. Find new home for loss

UnitGroupT = T.Literal["core", "optional"]
UnitStageT = T.Literal["load", "start", "step", "save", "update", "end"]
UnitStageDictT = dict[UnitStageT, list[TrainingUnit]]


@dataclass
class Units:
    """ Organizes all training units by lifecycle stage and category

    This dataclass acts as the central registry for all TrainingUnit instances participating in
    the training loop. It separates units into core (mandatory) and optional categories, then
    further groups them by their activation stage within the training lifecycle

    Core units always participate: LoadUnit, SaveUnit, PluginUnit, LossUnit, OptimizerUnit,
    EventUnit and StateUnit. Optional units are loaded conditionally based on configuration flags
    (e.g., TensorBoardUnit for logging, PreviewUnit for GUI previews)

    Units are organized into five lifecycle stages:
        - load       : Initialization phase after loading state from checkpoint
        - start      : Executed immediately prior to the first real training iteration
        - step       : Called once per training iteration after backward pass
        - save       : Triggers when checkpoints should be written to disk
        - update     : Signals GUI refresh or preview generation
        - end        : Final cleanup operations at training completion

    The Units object automatically tracks which units support each lifecycle stage via has_*
    properties and maintains a consolidated view of all active units in the .all property

    Parameters
    ----------
    This object is not expected to be instantiated directly. Instead, it is created and managed
    internally by the TrainingLoop class based on the configuration provided during initialization

    Notes
    ----
    The order of units within each stage matters. Core units always execute before optional units,
    except LoadUnit which intentionally runs last during the load phase to ensure all state loads
    first
    """
    stages_core: UnitStageDictT = field(default_factory=lambda: T.cast(UnitStageDictT,
                                                                       {"load": [],
                                                                        "start": [],
                                                                        "step": [],
                                                                        "save": [],
                                                                        "update": [],
                                                                        "end": []}))
    """ Dictionary mapping lifecycle stages to lists of core training units. Core units always
    participate regardless of configuration flags """

    stages_optional: UnitStageDictT = field(
        default_factory=lambda: T.cast(UnitStageDictT,
                                       {"load": [],
                                        "start": [],
                                        "step": [],
                                        "save": [],
                                        "update": [],
                                        "end": []})
        )
    """ Dictionary mapping lifecycle stages to lists of optional training units. These are loaded
    only if their corresponding configuration dictates that they are """

    _all: dict[UnitGroupT, dict[str, TrainingUnit]] | None = field(init=False, default=None)

    @property
    def all(self) -> dict[UnitGroupT, dict[str, TrainingUnit]]:
        """ Consolidated view of all units by group ("core" or "optional") and class name """
        if self._all is None:
            self._all = {}
            for key in T.cast(list[UnitGroupT], ["core", "optional"]):
                units = set(y for x in getattr(self, f"stages_{key}").values() for y in x)
                self._all[key] = {u.__class__.__name__: u for u in units}
            logger.debug("[Units] All units: %s", self._all)
        return self._all

    @property
    def core(self) -> dict[str, TrainingUnit]:
        """ Core training units grouped in a dictionary keyed by unit class names """
        return self.all["core"]

    @property
    def optional(self) -> dict[str, TrainingUnit]:
        """ Optional training units grouped in a dictionary keyed by unit class names """
        return self.all["optional"]

    @property
    def on_load(self) -> list[TrainingUnit]:
        """ Units registered for the load phase. Core units first, then optional. The LoadUnit is
        always last to ensure all units are configured before attempting to load state """
        load_unit = next(x for x in self.stages_core["load"] if isinstance(x, LoadUnit))
        core = [x for x in self.stages_core["load"] if x != load_unit]
        return core + self.stages_optional["load"] + [load_unit]

    @property
    def on_start(self) -> list[TrainingUnit]:
        """ Units registered for the start phase. Only optional units as core units can
        handle pre-training """
        return self.stages_optional["start"]

    @property
    def step(self) -> list[TrainingUnit]:
        """ All units registered for per-iteration callbacks """
        return self.stages_core["step"] + self.stages_optional["step"]

    @property
    def on_save(self) -> list[TrainingUnit]:
        """ All units registered for checkpoint saving operations (optional before core) """
        return self.stages_optional["save"] + self.stages_core["save"]

    @property
    def on_update(self) -> list[TrainingUnit]:
        """ All units registered for update/refresh events (optional before core) """
        return self.stages_optional["update"] + self.stages_core["update"]

    @property
    def on_end(self) -> list[TrainingUnit]:
        """ Units registered for final cleanup (optional before core) """
        return self.stages_optional["end"] + self.stages_core["end"]

    @property
    def have_state_dict(self) -> dict[str, TrainingUnit]:
        """ Subset of all units that implement a state_dict and can participate in checkpoint
        loading and saving """
        return {k: v for k, v in self.core.items() | self.optional.items()
                if v.has_state_dict}

    def add_unit(self, unit: TrainingUnit, is_core: bool) -> None:
        """ Register a training unit for appropriate lifecycle stages

        Automatically determines which lifecycle hooks the unit should participate in based on
        its has_* properties (has_start, has_step, has_save, etc.). Only registers units that
        declare support for each stage

        Parameters
        ----------
        unit
            The TrainingUnit instance to register
        is_core
            Whether this is a mandatory core unit or an optional enhancement unit

        Notes
        -----
        Units without any lifecycle stage capability (all has_* properties False) are silently
        ignored during registration
        """
        stage_group = self.stages_core if is_core else self.stages_optional
        for key in ("load", "start", "step", "save", "update", "end"):
            if not getattr(unit, f"has_{key}"):
                continue
            logger.debug("[Units] '%s' Adding 'stage_%s'['%s']",
                         unit.__class__.__name__, "core" if is_core else "optional", key)
            stage_group[key].append(unit)


class TrainStep:  # pylint:disable=too-many-instance-attributes
    """ Executes individual training iterations and manages the complete training lifecycle

    TrainStep orchestrates each iteration of the training process, coordinating all registered
    TrainingUnit instances through their respective lifecycle hooks. It handles:
        - Initialization phase (on_load) with model loading and state restoration
        - Pre-training setup (on_start) immediately prior to the first real iteration
        - Per-iteration work (step) including forward/backward passes via PluginUnit
        - Checkpoint management (save/update events from TrainingEvents)
        - Final cleanup (on_end) when training concludes

    The class manages the complete training workflow through a TrainStep object, which is itself
    wrapped in an FSThread within TrainingLoop to run asynchronously. All state including the
    model, optimizer, loader, and loss functions are contained within this instance

    Parameters
    ----------
    faceswap_model
        The FaceswapModel instance containing the neural network architecture and weights
    trainer
        TrainerPlugin instance executing forward/backward/optimization cycle
    loader
        TrainLoader providing input batches (images, targets, metadata)
    training_events
        TrainingEvents object enabling cross-thread communication with main process
    save_interval
        Number of iterations between automatic checkpoint saves. Default: 250
    snapshot_interval
        Number of iterations between snapshot creation for recovery points. Default: 25000

    Notes
    -----
    The iteration count is managed by StateUnit and automatically increments with each step().
    Current loss values are populated by PluginUnit.step() and detached to avoid computation graph
    issues
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
        """ The computational device (CPU or GPU) assigned for training operations """
        return self._device

    @property
    def events(self) -> TrainingEvents:
        """ Event system enabling signals to save/update requests and receive previews """
        return self._events

    @property
    def model(self) -> FaceswapModel:
        """ Reference to the loaded neural network with current weights and state """
        return self._model

    @property
    def units(self) -> Units:
        """ Registry of all registered training units organized by lifecycle stage """
        return self._units

    @property
    def optimizer_unit(self) -> OptimizerUnit:
        """ Optimizer managing learning rate, weight decay, gradient clipping, etc """
        return self._optimizer_unit

    @property
    def iteration(self) -> int:
        """ Current total training iteration number (exposed via StateUnit). Read-only """
        return T.cast(StateUnit, self._units.core["StateUnit"]).iteration

    @property
    def session_iteration(self) -> int:
        """ Current session training iteration number (exposed via StateUnit). Read-only """
        return T.cast(StateUnit, self._units.core["StateUnit"]).session_iteration

    @property
    def current_loss(self) -> list[BatchLoss]:
        """ List of computed loss values for the most recent batch. Read-only """
        return T.cast(PluginUnit, self._units.core["PluginUnit"]).current_loss

    def _get_optimizer(self, model: ModelPlugin) -> OptimizerUnit:
        """ Initialize and configure the optimizer unit

        Parameters
        ----------
        model
            The ModelPlugin instance containing trainable parameters

        Returns
        -------
        Configured optimizer unit ready for training iterations
        """
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
        """ Initialize and configure the plugin unit for training operations

        Parameters
        ----------
        loader
            TrainLoader providing input batches (images, masks, targets, metadata)
        trainer
            TrainerPlugin executing forward/backward/optimization cycle
        optimizer
            OptimizerUnit managing parameter updates during training

        Returns
        -------
        Configured plugin unit ready for iteration execution
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
        """ Initialize core training units required for any training session

        Parameters
        ----------
        loader
            TrainLoader for input data batches
        trainer
            TrainerPlugin executing training operations
        events
            TrainingEvents for cross-thread communication
        save_interval
            Number of iterations between automatic checkpoint saves
        snapshot_interval
            Number of iterations between snapshot creation (0 = disabled)
        """
        save_train_state = T.cast(T.Literal["always", "never", "exit"],
                                  mod_cfg.Optimizer.save_optimizer())

        plugin_unit = self._get_plugin_unit(loader, trainer, self.optimizer_unit)
        loss_unit = LossUnit(mod_cfg.nan_protection(), plugin_unit.current_loss, self._device)
        save_unit = SaveUnit(self.model,
                             self.optimizer_unit,
                             events,
                             loss_unit.current_average,
                             save_interval,
                             snapshot_interval,
                             save_train_state)

        self._units.add_unit(StateUnit(self._model.state, trainer.batch_size), is_core=True)
        self._units.add_unit(plugin_unit, is_core=True)
        self._units.add_unit(self.optimizer_unit, is_core=True)
        self._units.add_unit(loss_unit, is_core=True)
        self._units.add_unit(EventUnit(events), is_core=True)
        self._units.add_unit(LoadUnit(self.model), is_core=True)
        self._units.add_unit(save_unit, is_core=True)

    def add_unit(self, unit: TrainingUnit) -> None:
        """ Register an optional training unit.

        Adds a non-core unit to the units registry for the appropriate lifecycle stages.

        Parameters
        ----------
        unit
            The TrainingUnit instance to register as optional.

        Notes
        -----
        This method is typically called from TrainingLoop.add_unit() rather than directly.
        """
        self._units.add_unit(unit, is_core=False)

    def _on_loop_start(self) -> None:
        """ Execute initialization phase for all registered units """
        logger.debug("[TrainStep] Loading")
        for unit in self._units.on_load:
            logger.debug("[TrainStep] Executing on_load: '%s'", unit.__class__.__name__)
            unit.on_load(self)

        if self.iteration < 0:
            logger.debug("[TrainStep] Entering pre-train")
        self._started = True

    def _on_train_start(self) -> None:
        """ Execute actions prior to real training for all registered units """
        logger.debug("[TrainStep] Starting main train")
        for unit in self.units.on_start:
            unit.on_start()

    def _step(self) -> None:
        """ Execute one training iteration step for all units """
        for unit in self._units.step:
            logger.trace("[TrainStep] %s step %s",  # type:ignore[attr-defined]
                         unit.__class__.__name__, self.iteration)
            unit.step(self.iteration)

    def _save(self) -> None:
        """ Execute save operations for all registered units """
        for unit in self._units.on_save:
            logger.debug("[TrainStep] %s Saving step %s", unit.__class__.__name__, self.iteration)
            unit.on_save(self.iteration)
        self._events.save.clear()

    def _update(self) -> None:
        """ Execute update operations for all registered units """
        for unit in self._units.on_update:
            logger.debug("[TrainStep] %s Updating", unit.__class__.__name__)
            unit.on_update()
        self._events.update.clear()

    def step(self) -> None:
        """ Execute one complete training iteration

        Orchestrates the full training loop for a single batch:
            1. If not started, triggers initialization phase (on_load hooks)
            2. Executes per-iteration work (step hooks via core and optional TrainingUnit)
            3. Checks for save request event and saves if needed
            4. Checks for update request event and updates preview if needed

        Notes
        -----
        The first call to step() automatically initializes the training loop. Subsequent calls
        assume proper initialization has occurred
        """
        if not self._started:
            self._on_loop_start()

        if self.iteration == 0:
            self._on_train_start()

        logger.trace("[TrainStep] step %s",  self.iteration)  # type:ignore[attr-defined]
        self._step()

        if self._events.save.is_set():
            self._save()

        if self._events.update.is_set():
            self._update()

    def on_end(self) -> None:
        """ Execute final cleanup operations when training concludes

        Calls on_end() hook for each unit in the end lifecycle group. This ensures all units
        perform necessary cleanup like releasing GPU memory, closing file handles, or finalizing
        logging before training terminates
        """
        for unit in self._units.on_end:
            logger.trace("[TrainStep] ending %s",  # type:ignore[attr-defined]
                         unit.__class__.__name__)
            unit.on_end()
        logger.debug("[TrainStep] Training ended")


@dataclass
class TrainingEvents:
    """ Event system enabling communication between training thread and main process

    This dataclass provides a synchronized event mechanism for coordinating operations across
    threads. The training loop runs in a background FSThread, while the monitor runs on the main
    thread. Events allow the monitor to request checkpoints (save), preview updates (update),
    mask toggles, or exit training without blocking the training loop

    Thread Safety:
    -----------
    All event operations are protected by a threading.Lock preventing race conditions when
    accessing preview (shared between threads). The lock ensures atomic reads/writes of the
    preview data which can be large numpy arrays

    Preview Support:
    ------------
    The training preview is held internally in this object. `get_preview()` retrieves and clears
    this atomically, while `set_preview()` stores previews for the next iteration to retrieve from
    the main loop
    """
    save: Event = field(default_factory=Event)
    """ Event object signaling checkpoint saving request from the main thread """
    exit: Event = field(default_factory=Event)
    """ Event object requesting immediate training termination (after save) """
    update: Event = field(default_factory=Event)
    """ Event object requesting preview refresh or status update """
    toggle_mask: Event = field(default_factory=Event)
    """ Event object for mask inversion operations during training visualization """
    _preview: None | npt.NDArray[np.uint8] = None
    _lock: Lock = field(default_factory=Lock)

    def get_preview(self) -> None | npt.NDArray[np.uint8]:
        """ Retrieve and clear the latest preview image generated during training

        Atomically reads and clears the preview buffer to prevent multiple calls from blocking each
        other. Subsequent calls will return None until set_preview() is called with a new preview
        in the training thread

        Returns
        -------
        The latest preview image (BGR, uint8, (H, W, C)) or ``None`` if no preview available
        """
        with self._lock:
            if self._preview is None:
                return None
            retval = self._preview
            self._preview = None
        logger.debug("[TrainingEvents] Getting preview: %s", retval.shape)
        return retval

    def set_preview(self, preview: npt.NDArray[np.uint8]) -> None:
        """ Store a preview image for retrieval by next `get_preview()` call

        Called by the training thread to render a preview image for display by the main thread

        Parameters
        ----------
        preview
            The latest preview image (BGR, uint8, (H, W, C))
        """
        logger.debug("[TrainingEvents] Setting preview: %s", preview.shape)
        with self._lock:
            self._preview = preview


class TrainingLoop:
    """ Main training loop orchestrator that runs model training in a background thread

    TrainingLoop is the primary interface for initiating and controlling training sessions.
    It wraps TrainStep functionality within an FSThread, allowing the training process to run
    asynchronously without blocking the main GUI thread. The loop continues until either:
        - All configured iterations complete
        - Exit event is set via the main thread (KeyboardInterrupt or button click)

    Threading Architecture:
    -------------------
    TrainingLoop creates an FSThread wrapping the main training loop. This separation allows:
        - The main thread to call loop.start() and continue responding to user input
        - KeyboardInterrupt from GUI to be caught and handled gracefully
        - Preview updates via set_preview/get_preview to work across threads safely

    Unit Registration:
    --------------
    Units can be added in two ways:
        - During TrainStep initialization: Core units (always loaded) are registered automatically
          based on configuration flags (save_interval, snapshot_interval)
        - Via `TrainingLoop.add_unit()`: Optional units loaded from config fileif their
        corresponding flags are enabled (e.g., TensorBoardUnit if tb_logging=True)

    Lifecycle Flow:
    ------------
    1. start() : Begins FSThread execution of _main_loop()
    2. Loop runs iterations calling stepper.step() each time:
       a. Initialize via on_load hooks (LoadUnit loads checkpoint state first)
       b. Handle pre-train to train transition via in_start hooks
       c. Execute iteration via step hooks (PluginUnit.forward/backward/optimizer)
       d. Save checkpoints if requested or at intervals
       e. Update GUI if preview available or update requested
    3. on_end() : Called when iterations complete or exit requested, runs cleanup hooks

    Parameters
    ----------
    iterations
        Total number of training iterations to perform before stopping automatically.
        Loop terminates early if exit event is set via main thread interrupt
    faceswap_model
        FaceswapModel instance containing neural network architecture and weights
    trainer
        TrainerPlugin executing forward/backward/optimization cycle per iteration
    loader
        TrainLoader providing input batches (images, masks, targets, metadata)
    save_interval
        Number of iterations between automatic checkpoint saves. Default: 250
    snapshot_interval
        Number of iterations between snapshot creation for recovery points. Default: 25000

    Notes
    ----
    The TrainingLoop creates all core units during __init__ via TrainStep, but optional units
    are registered separately. This allows dynamic unit loading based on runtime configuration

    Keyboard interrupts caught in _main_loop() trigger graceful shutdown by calling
    stepper.on_end() which performs final saves and cleanup operations before terminating the
    thread.
    """
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
        """ Event system enabling GUI to request saves, previews, mask toggles, or exit """
        return self._events

    def check_and_re_raise_error(self) -> None:
        """ Check and propagate any errors that occurred during background training execution.

        Called periodically from the main thread to check if the training thread encountered an
        exception. If an error was raised in the training thread, this re-raises it so the main
        thread can display appropriate error messages and halt operations
        """
        self._thread.check_and_raise_error()

    def _training_loop(self) -> None:
        """ Execute the core training iteration loop """
        while True:
            if self._stepper.session_iteration == self._iterations - 1:
                logger.debug("[TrainingLoop] Total iterations reached. Signalling exit iter")
                self._events.exit.set()
            self._stepper.step()
            if self._events.exit.is_set():
                logger.debug("[TrainingLoop] Exit requested")
                break

        self._stepper.on_end()
        logger.debug("[TrainingLoop] Training Complete")

    def _main_loop(self) -> None:
        """ Wrap the training loop to handle KeyboardInterrupt and start the loop """
        logger.debug("[TrainingLoop] Commencing Training")
        try:
            self._training_loop()
        except KeyboardInterrupt:
            try:
                logger.info("[Train] Keyboard Interrupt Caught. Saving Weights and exiting")  # TODO check location
                self._stepper.on_end()
            except KeyboardInterrupt:
                logger.warning("Saving model weights has been cancelled!")
        except Exception as err:
            raise err

    def add_unit(self, unit: TrainingUnit) -> None:
        """ Register an optional training unit for the session.

        Adds a non-core unit to be executed during training iterations. Units can be added
        dynamically based on runtime configuration or user requests from main thread

        Parameters
        ----------
        unit
            The TrainingUnit instance to register as an optional unit
        """
        self._stepper.add_unit(unit)

    def start(self) -> None:
        """ Begin training by starting the background thread """
        self._thread.start()

    def join(self) -> None:
        """ Wait for training to complete by joining the background thread """
        self._thread.join()


__all__ = get_module_objects(__name__)

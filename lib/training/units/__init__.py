#! /usr/bin/env python3
""" Training units - modular building blocks that extend the training loop lifecycle

Units are specialized components that plug into the TrainingLoop at specific lifecycle points
(Initialization, per-iteration work, checkpoint saves, periodic updates, final cleanup). Each unit
inherits from TrainingUnit and implements only the hooks it needs, making them focused and
efficient

Organization by function:

Core units (mandatory)
----------------------
    Always loaded automatically - form the essential backbone for any training session

    Located in lib.training.units.core
        TrainingUnit     : Base class defining lifecycle contract
        EventUnit       : Handles communication between components
        LossUnit        : Computes loss metrics per batch
        OptimizerUnit   : Manages gradient updates and learning rates
        PluginUnit      : Executes model inference pass
        SaveUnit        : Handles checkpoint persistence operations
        StateUnit       : Tracks training state transitions

Optional units (conditional)
------------------------
    Loaded only when config flags enable them - provide enhancement features

    Located in lib.training.units
        TensorBoardUnit : Logs metrics to visualization dashboard
        PreviewUnit     : Generates live preview images during training
        TimelapseUnit   : Saves timelapse recordings at checkpoints
        WarmupUnit      : Implements learning rate warmup scheduling
        LRFinderUnit    : Performs learning rate finding before training

Usage pattern: TrainingLoop collects all units, checks their capabilities via has_* properties,
and calls corresponding methods (on_load, on_start, step, on_save) at appropriate times in the
lifecycle
"""
from .core import TrainingUnit
from .lr_finder_unit import LRFinderUnit
from .preview_unit import PreviewUnit, TimelapseUnit
from .tensorboard_unit import TensorBoardUnit
from .warmup_unit import WarmupUnit
from .weights_unit import FreezeWeightsUnit, LoadWeightsUnit

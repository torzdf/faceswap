#! /usr/bin/env python3
""" Core units are mandatory components required by every training session

This package contains the essential building blocks that all training loops use automatically:

- TrainingUnit      : Abstract base class defining the lifecycle contract (on_start, step, etc.)
- EventUnit         : Handles event flags and signals between training components
- LossUnit          : Computes loss values for each batch during forward/backward passes
- OptimizerUnit     : Manages optimizer state and learning rate updates after gradient computation
- PluginUnit        : Executes model inference (forward pass) through the neural network
- SaveUnit          : Handles checkpoint saving, loading, and snapshot creation operations
- StateUnit         : Manages training state transitions (pre_training -> training -> completed)

These units are always loaded when TrainingLoop initializes - they form the mandatory backbone
of every training session. Unlike optional units like TensorBoardUnit or PreviewUnit which can be
disabled in configuration, these core units must exist for basic training to function at all.
"""
from .base import TrainingUnit
from .event_unit import EventUnit
from .loss_unit import LossUnit
from .optimizer_unit import OptimizerUnit
from .plugin_unit import PluginUnit
from .save_unit import LoadUnit, SaveUnit
from .state_unit import StateUnit

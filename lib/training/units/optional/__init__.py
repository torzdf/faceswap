#! /usr/bin/env python3
""" Optional training units that extend functionality based on user configuration.

This package contains conditional components that enhance the base training loop with monitoring
and analysis features:

- TensorBoardUnit   : Logs metrics to TensorBoard for visualization
- PreviewUnit       : Generates live preview images during active training sessions  
- TimelapseUnit     : Creates timelapse recordings at checkpoint save intervals  
- WarmupUnit        : Implements learning rate warmup scheduling for stable initial training  
- LRFinderUnit      : Performs learning rate finding before main training begins  

These units are loaded only when their corresponding config flags are set to True. They work 
alongside the core units (from lib.training.units.core) but can be disabled independently, making 
them perfect for optional features like live preview generation, TensorBoard monitoring, or 
learning rate analysis tools that aren't essential but useful when enabled
"""
from .lr_finder_unit import LRFinderUnit
from .preview_unit import PreviewUnit, TimelapseUnit
from .tensorboard_unit import TensorBoardUnit
from .warmup_unit import WarmupUnit

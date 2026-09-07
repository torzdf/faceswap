#!/usr/bin/env python3
""" Loading and preparation of data for training FaceSwap models

This package sits between the extracted face folders and the training loop, exposing the
dataset classes, batch collation helpers, and PyTorch-backed loaders that feed the model
both while training and when generating previews. At import time it exposes these names

- ``get_label``: per-side label for a dataset index, cycling through the identity alphabet (A-Z,
0-9, a-z) starting at A
- :class:`BatchMeta`: dataclass holding the face, eye and mouth mask tensors required to train a
batch of images
- :class:`PreviewLoader`: iterator that yields preview batches (inputs and targets) per side for
previews and evaluation
- :class:`TrainLoader`: builds per-side datasets from validated folders and wraps them in PyTorch
loaders that collate ready-to-feed training batches
"""
from .data_set import get_label
from .collate import BatchMeta
from .loader import PreviewLoader, TrainLoader

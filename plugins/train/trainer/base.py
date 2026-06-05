#!/usr/bin/env python3
"""Base Class for Faceswap Trainer plugins. All Trainer plugins should be inherited from
this class.

At present there is only the :class:`~plugins.train.trainer.original` plugin, so that entirely
inherits from this class. If further plugins are developed, then common code should be kept here,
with "original" unique code split out to the original plugin.
"""
from __future__ import annotations
import abc
import logging
import typing as T
from dataclasses import dataclass
from contextlib import nullcontext

import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects


if T.TYPE_CHECKING:
    from lib.training.data import BatchMeta
    from lib.training.loss import LossCollator, BatchLoss
    from lib.training.optimizer import Optimizer
    from plugins.train.model.base import ModelPlugin

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:  # pylint:disable=too-many-instance-attributes
    """Configuration for training a model

    Parameters
    ----------
    image_folders
        List of folders to be used as inputs to the model. Folders are provided in processing order
        (eg: [A, B, ...])
    model_folder
        The folder that the model is to be loaded/saved to or from
    batch_size
        The batch size to load data from each of the loaders
    warmup_steps
        The number of steps to warm the learning rate up for
    augment_color
        ``True`` to perform color augmentation otherwise ``False``
    flip
        ``True`` to perform image flipping otherwise ``False``
    warp
        ``False`` to disable warping ``True`` to enable warping
    cache_landmarks
        ``True`` to cache landmarks from the other side for Warp to landmarks
    use_lr_finder
        ``True`` to use the learning rate finder. Default: ``False``
    snapshot interval
        The number of iterations between snapshots. Default -1 (Disabled)
    """
    folders: list[str]
    """List of folders to be used as inputs to the model. Folders are provided in processing order
    (eg: [A, B, ...])"""
    model_folder: str
    """The folder that the model is to be loaded/saved to or from"""
    batch_size: int
    """The batch size to load data from each of the loaders"""
    warmup_steps: int
    """The number of steps to warm the learning rate up for"""
    augment_color: bool
    """``True`` to perform color augmentation otherwise ``False``"""
    flip: bool
    """``False`` to disable warping ``True`` to enable warping"""
    warp: bool
    """``False`` to disable warping ``True`` to enable warping"""
    no_logs: bool
    """``True`` to disable Tensorboard logging, ``False`` to enable"""
    cache_landmarks: bool
    """``True`` to cache landmarks from the other side for Warp to landmarks"""
    lr_finder: bool = False
    """``True`` to use the learning rate finder"""
    snapshot_interval: int = -1
    """The number of iterations between snapshots"""


class TrainerBase(abc.ABC):
    """A trainer plugin interface. It must implement the method "train_batch" which takes an input
    of inputs to the model and target images for model output. It returns loss per side

    Parameters
    ----------
    model
        The configured Faceswap model plugin to be trained
    batch_size
        The batch size to train the model at
    mixed_precision
        ``True`` to enable mixed precision training. ``False`` for float32
    device_type
        The torch device type that is training the model
    """
    def __init__(self,
                 model: ModelPlugin,
                 batch_size: int,
                 mixed_precision: bool,
                 device_type: str) -> None:
        logger.debug(parse_class_init(locals()))
        self.model: ModelPlugin = model
        """The model plugin to be trained"""
        self.batch_size = batch_size
        """The batch size for each iteration to be trained through the model."""
        self.sampler = self.get_sampler()
        """The data sampler that the data loader should use"""
        self._forward_context = (torch.autocast(device_type=device_type, dtype=torch.float16)
                                 if mixed_precision else nullcontext())

    @abc.abstractmethod
    def get_sampler(self) -> type[torch.utils.data.RandomSampler |
                                  torch.utils.data.DistributedSampler]:
        """Override to set the sampler that the Torch DataLoader should use

        Returns
        -------
        The sampler that the torch DataLoader should use
        """

    @abc.abstractmethod
    def backward(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
        """Override to run a single backward pass through the model for a single batch

        Parameters
        ----------
        loss
            The loss scalar to use for backprop
        optimizer
            The configured optimizer to use for backprop
        """

    @abc.abstractmethod
    def forward(self,
                inputs: list[torch.Tensor],
                targets: list[torch.Tensor],
                meta: BatchMeta,
                loss_func: LossCollator) -> list[BatchLoss]:
        """Override to run a single forward pass through the model for a single batch

        Parameters
        ----------
        inputs
            The batch of input image tensors to the model of length(num inputs)
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch
        loss_func
            The configured loss functions to use
        """

    def __call__(self,
                 inputs: list[torch.Tensor],
                 targets: list[torch.Tensor],
                 meta: BatchMeta,
                 loss_func: LossCollator,
                 optimizer: Optimizer) -> list[BatchLoss]:
        """Runs the plugin's forward and backwards passed through the model for a single batch

        Parameters
        ----------
        inputs
            The batch of input image tensors to the model of length(num inputs)
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch
        loss_func
            The selected loss functions for the model
        optimizer
            The configured Optimizer to use

        Returns
        -------
        The loss for each input to the model in order (A, B, ...)
        """
        with self._forward_context:
            loss = self.forward(inputs, targets, meta, loss_func)
        logger.trace("[%s] Losses: %s", self.__class__.__name__, loss)  # type:ignore[attr-defined]
        total_loss = T.cast(torch.Tensor, sum(x.total for x in loss))
        self.backward(total_loss, optimizer)
        return loss


__all__ = get_module_objects(__name__)

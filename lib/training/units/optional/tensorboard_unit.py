#!/usr/bin/env python3
""" TensorBoard logging unit for training monitoring.

This optional module provides the TensorBoardUnit class which handles logging training metrics to
TensorBoard for visualization and analysis. It supports writing model graphs during initialization,
logging various loss components, and managing the lifecycle of the TensorBoard writer throughout
the training process.

The unit integrates with the Faceswap training loop system and can optionally write model
architecture diagrams for better understanding of network structure. It handles both regular and
live log file reading through the RecordIterator helper class
"""
from __future__ import annotations

import logging
import os
import struct
import typing as T

import torch
from torch.utils.tensorboard import SummaryWriter

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from lib.training.data import get_label
from lib.training.units.core import TrainingUnit

if T.TYPE_CHECKING:
    from lib.training.loss import BatchLoss
    from plugins.train.model.base import ModelPlugin
    from lib.training.training_loop import TrainStep

logger = logging.getLogger(__name__)


class RecordIterator:
    """ Iterator for reading TensorBoard event files

    This iterator reads TensorFlow event records from a TensorBoard log file, handling both regular
    and live (continuously updated) files. It properly manages file positioning and handles partial
    reads or corrupted data gracefully

    Parameters
    ----------
    log_file
        Path to the TensorBoard event log file to read from
    is_live, optional
        ``True`` if this is a live log file that may still be written to. Default: ``False``
    """
    _max_record_size = 1024 ** 3

    def __init__(self, log_file, is_live: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        self._file_path = log_file
        self._log_file = open(self._file_path, "rb")  # pylint:disable=consider-using-with
        self._is_live = is_live
        self._position = 0
        logger.debug("Initialized %s", self.__class__.__name__)

    def __iter__(self) -> RecordIterator:
        """ Return the iterator object itself """
        return self

    def _on_file_read(self) -> None:
        """ Handle file operations when reading data """
        if not self._is_live or not self._log_file.closed:
            return

        logger.trace("Re-opening '%s' and Seeking to %s",  # type:ignore[attr-defined]
                     self._file_path, self._position)
        self._log_file = open(self._file_path, "rb")  # pylint:disable=consider-using-with
        self._log_file.seek(self._position, 0)

    def _on_file_end(self) -> None:
        """ Handle cleanup when reaching end of file """
        if self._is_live:
            self._position = self._log_file.tell()
            logger.trace("Setting live position to %s",  # type:ignore[attr-defined]
                         self._position)

        logger.trace("EOF. Closing '%s'", self._file_path)  # type:ignore[attr-defined]
        self._log_file.close()

    def __next__(self) -> bytes:
        """ Return the next record from the log file

        Returns
        -------
        The raw event record data

        Raises
        ------
        StopIteration
            When end of file is reached or partial read occurs
        """
        self._on_file_read()

        record_start = self._log_file.tell()
        b_header = self._log_file.read(8)

        if len(b_header) < 8:  # Partial header. Rewind for next call
            self._log_file.seek(record_start, 0)
            self._on_file_end()
            raise StopIteration

        read_len = int(struct.unpack('Q', b_header)[0])
        if read_len > self._max_record_size:
            logger.debug("Implausible record length %s in '%s' at offset %s; treating as partial "
                         "and stopping.", read_len, self._file_path, record_start)
            self._log_file.seek(record_start, 0)
            self._on_file_end()
            raise StopIteration

        len_crc = self._log_file.read(4)
        data = self._log_file.read(read_len)
        data_crc = self._log_file.read(4)
        if len(len_crc) < 4 or len(data) < read_len or len(data_crc) < 4:  # Partial read
            self._log_file.seek(record_start, 0)
            self._on_file_end()
            raise StopIteration

        logger.trace("Returning event data of len %s", read_len)  # type:ignore[attr-defined]

        return data


class TensorBoardUnit(TrainingUnit):
    """ TensorBoard logging unit for training monitoring

    This unit handles logging training metrics to TensorBoard for visualization and analysis. It
    can optionally write model graphs during the initial setup phase, logs various loss components,
    and manages the lifecycle of the TensorBoard writer throughout the training process

    Parameters
    ----------
    model_folder
        Path to the folder where model files are stored
    model_name
        Name identifier for this model
    session_id
        Unique identifier for the current training session
    write_graph, optional
        Whether to write the model graph during initialization. Default: ``True``
    """
    def __init__(self,
                 model_folder: str,
                 model_name: str,
                 session_id: int,
                 write_graph: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._model_folder = model_folder
        self._model_name = model_name
        self._session_id = session_id
        self._write_graph = write_graph
        log_dir = os.path.join(model_folder,
                               f"{model_name}_logs",
                               f"session_{session_id}",
                               "train")
        logger.debug("%s Logging to: '%s'", self.log_name, log_dir)
        self._writer = SummaryWriter(log_dir)

        self._current_loss: list[BatchLoss]  # set in on_load

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = ", ".join(f"{k[1:]}={v!r}" for k, v in self.__dict__.items()
                           if k in ("_model_folder",
                                    "_model_name",
                                    "_session_id",
                                    "_write_graph"))
        return f"{self.__class__.__name__}({params})"

    def _write_torch_graph(self,
                           model: ModelPlugin,
                           device: torch.Device,
                           input_shapes: list[tuple[int, int, int]]) -> None:
        """ Write the PyTorch model graph to TensorBoard

        Parameters
        ----------
        model
            The model plugin whose graph will be written
        device
            Device on which the model is located
        input_shapes
            Input shapes for creating dummy inputs for graph tracing [(C, H, W), ]
        """
        logger.debug("%s Writing model graph: %s", self.log_name, model)
        is_training = model.training
        model.eval()
        with torch.no_grad():
            inputs = tuple(torch.rand((1, *shape)).to(device) for shape in input_shapes)
            self._writer.add_graph(model, (inputs, ))
        if is_training:
            model.train()

    def on_load(self, loop: TrainStep) -> None:
        """ Initialize TensorBoard logging and write model graph

        Sets up the TensorBoard writer with appropriate log directory and writes the model
        architecture graph for visualization during the first session

        Parameters
        ----------
        loop
            The training step object managing this unit's lifecycle
        """
        self._current_loss = loop.current_loss
        model = loop.model
        logger.debug("%s Setting up TensorBoard Logging", self.log_name)
        if self._session_id == 1 and self._write_graph:
            self._write_torch_graph(model.plugin, loop.device, model.info.input_shapes)
        logger.verbose("%s TensorBoard logging Enabled",  # type: ignore[attr-defined]
                       self.log_name)

    def _get_logs(self,
                  loss: list[BatchLoss]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """ Extract and format loss metrics for logging

        Parameters
        ----------
        loss
            List of batch loss objects to extract metrics from

        Returns
        -------
        Dictionary mapping metric names to their values
        """
        retval: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "total": T.cast(torch.Tensor, sum(x.total for x in loss))}
        for i, out in enumerate(loss):
            lbl = get_label(i, len(loss))
            for idx, (w, u) in enumerate(zip(out.weighted, out.unweighted)):
                key = lbl if len(out.unweighted) == 1 else f"{lbl}_{idx}"
                weighted = {k: v.mean() for k, v in w.items()}
                unweighted = {k: v.mean() for k, v in u.items()}
                retval[f"face_{key}"] = T.cast(torch.Tensor, sum(weighted.values()))
                retval[f"weighted_{key}"] = weighted
                retval[f"unweighted_{key}"] = unweighted
            if out.mask is not None:
                retval[f"mask_{lbl}"] = out.mask.mean()
        return retval

    def step(self, iteration: int) -> None:
        """ Log batch metrics to TensorBoard

        Processes the current loss values and logs them to TensorBoard as scalar values.
        Skips logging during pre-training phase (negative iterations).

        Parameters
        ----------
        iteration
            Current training iteration number. Negative values indicate pre-training phase
        """
        if iteration < 0:
            logger.trace("%s Pre-training. Not handling Tensorboard",  # type:ignore[attr-defined]
                         self.log_name)
            return
        logs = self._get_logs(self._current_loss)
        logger.trace("%s Extracted logs [%s]: %s",  # type:ignore[attr-defined]
                     self.log_name, iteration, logs)

        for key, value in logs.items():
            tag = f"batch_{key}"
            if isinstance(value, torch.Tensor):
                self._writer.add_scalar(tag, value, global_step=iteration)
            elif isinstance(value, dict):
                for k, v in value.items():
                    self._writer.add_scalar(f"{tag}/{k}", v, global_step=iteration)
            else:
                raise ValueError(f"Unhandled Tensorboard data: {key}: {value}")

    def on_save(self, iteration: int) -> None:
        """ Flush the TensorBoard writer

        Forces writing all pending logs to disk at save intervals

        Parameters
        ----------
        iteration
            Current training iteration number when saving occurs
        """
        logger.debug("%s Flushing Tensorboard writer [%s]", self.log_name, iteration)
        self._writer.flush()

    def on_end(self) -> None:
        """ Close the TensorBoard writer

        Flushes and closes the TensorBoard writer at the end of training.
        """
        logger.debug("%s Exiting Tensorboard writer", self.log_name)
        self._writer.flush()
        self._writer.close()


__all__ = get_module_objects(__name__)

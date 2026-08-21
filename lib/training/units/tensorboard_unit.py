#!/usr/bin/env python3
""" Tensorboard call back for PyTorch logging. """
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
from .core import TrainingUnit

if T.TYPE_CHECKING:
    from lib.training.loss import BatchLoss
    from plugins.train.model.base import ModelPlugin
    from lib.training.training_loop import TrainStep

logger = logging.getLogger(__name__)


class RecordIterator:
    """A replacement for tensorflow's :func:`compat.v1.io.tf_record_iterator`

    Parameters
    ----------
    log_file
        The event log file to obtain records from
    is_live
        ``True`` if the log file is for a live training session that will constantly provide data.
        Default: ``False``
    """
    _max_record_size = 1024 ** 3
    """Maximum size for a TFRecord. Caps at 1GB to protect against nonsense length bytes"""

    def __init__(self, log_file, is_live: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        self._file_path = log_file
        self._log_file = open(self._file_path, "rb")  # pylint:disable=consider-using-with
        self._is_live = is_live
        self._position = 0
        logger.debug("Initialized %s", self.__class__.__name__)

    def __iter__(self) -> RecordIterator:
        """Iterate over a Tensorboard event file"""
        return self

    def _on_file_read(self) -> None:
        """If the file is closed and we are reading live data, re-open the file and seek to the
        correct position"""
        if not self._is_live or not self._log_file.closed:
            return

        logger.trace("Re-opening '%s' and Seeking to %s",  # type:ignore[attr-defined]
                     self._file_path, self._position)
        self._log_file = open(self._file_path, "rb")  # pylint:disable=consider-using-with
        self._log_file.seek(self._position, 0)

    def _on_file_end(self) -> None:
        """Close the event file. If live data, record the current position"""
        if self._is_live:
            self._position = self._log_file.tell()
            logger.trace("Setting live position to %s",  # type:ignore[attr-defined]
                         self._position)

        logger.trace("EOF. Closing '%s'", self._file_path)  # type:ignore[attr-defined]
        self._log_file.close()

    def __next__(self) -> bytes:
        """Get the next event log from a Tensorboard event file

        Returns
        -------
        A Tensorboard event log

        Raises
        ------
        StopIteration
            When the event log is fully consumed
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
    """ A unit for updating logs required to enable visualizations in TensorBoard.

    Parameters
    ----------
    model_folder
        The path of the directory where the model files are being saved to
    model_name
        The name of the model being trained on
    session_id
        The training session id that is about to commence
    write_graph
        Whether to visualize the graph in TensorBoard. Note that the log file can become quite
        large when `write_graph` is set to `True`. Default: ``True``
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

        self._current_loss: list[BatchLoss]  # set in on_start

    def __repr__(self) -> str:
        """ String representation for debugging and logging """
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
        """ Writes Faceswap model graph network to TensorBoard.

        Parameters
        ----------
        model
            The Faceswap model plugin to trace
        device
            The device the model is training on
        input_shapes
            The shape of the inputs to the model [(C, H, W), ]
        """
        logger.debug("%s Writing model graph: %s", self.log_name, model)
        is_training = model.training
        model.eval()
        with torch.no_grad():
            inputs = tuple(torch.rand((1, *shape)).to(device) for shape in input_shapes)
            self._writer.add_graph(model, (inputs, ), use_strict_trace=True)  # TODO to False
        if is_training:
            model.train()

    def on_start(self, loop: TrainStep) -> None:
        """ Create a new log file on session start. Write the graph if this is a new model

        Parameters
        ----------
        trainer
            The configured trainer object that is about to process it's first batch
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
        """ Extract the TensorBoard logs from the current batch loss items

        Parameters
        ----------
        loss
            A list of the current batch losses

        Returns
        -------
        The loss for the current batch formatted for TensorBoard logging
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
        """ Update Tensorboard logs on batch end

        Parameters
        ----------
        iteration
            The current iteration count
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
        """Flush data to disk on save

        Parameters
        ----------
        iteration
            The total iteration number for the model
        """
        logger.debug("%s Flushing Tensorboard writer [%s]", self.log_name, iteration)
        self._writer.flush()

    def on_end(self) -> None:
        """Close the writer on train completion

        Parameters
        ----------
        logs
            Unused
        """
        logger.debug("%s Exiting Tensorboard writer", self.log_name)
        self._writer.flush()
        self._writer.close()


__all__ = get_module_objects(__name__)

#! /usr/env/bin/python3
"""Interfaces for Faceswap extract plugins"""
from __future__ import annotations

import abc
import logging
import typing as T
from threading import Event

import numpy as np
import numpy.typing as npt
import torch

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from lib.align.constants import CenteringType


logger = logging.getLogger(__name__)


class ExtractPlugin(abc.ABC):
    """Base extract plugin that all plugins must inherit from.

    Parameters
    ----------
    input_size
        The size of the input required by the plugin. The input will always be square at these
        dimensions
    batch_size
        The batch size that the plugin processes data at. Note: Only the `process` method is
        guaranteed to receive data at this batch size (or less). The other processes may receive
        higher batch sizes for re-processing reasons. Do not rely on this when processing data.
        Default: `1`
    is_rgb
        ``True`` if the plugin expects input images to be RGB rather than BGR. Default: ``False``
    dtype
        A valid datatype that the plugin expects to receive the image at. Default: "float32"
    scale
        The scale that the plugin expects to receive the image at eg: (0, 255) for uint8 images.
        Default: (0, 1)
    force_cpu
        For Torch models, force running on the CPU, rather than the accelerated device. Sets the
        :class:`torch.device` to :attr:`device`. Default: ``False``
    """
    _compile_warning = Event()

    def __init__(self,
                 input_size: int,
                 batch_size: int = 1,
                 is_rgb: bool = False,
                 dtype: str = "float32",
                 scale: tuple[int, int] = (0, 1),
                 force_cpu: bool = False) -> None:
        self.input_size = input_size
        """The size of the plugin's input in pixels"""
        self.name = self.__class__.__name__
        """The name of the plugin. Derived from the module name"""
        self.batch_size = batch_size
        """The maximum batch size that this plugin's 'process' method will receive"""
        self.is_rgb = is_rgb
        """``True`` if the plugin expects RGB images. ``False`` for BGR"""
        self.dtype = dtype
        """The datatype that the plugin expects images at"""
        self.scale = scale
        """The numeric range that the plugin expects images to be in"""
        self.device = self._get_device(force_cpu)
        """The selected device to run torch ops on"""
        self.compile: bool = False
        """Set by the runner calling this plugin. Set to ``True`` if the plugin should compile
        any Torch models"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {k: v for k, v in self.__dict__.items()
                  if k in ["input_size", "batch_size", "is_rgb", "dtype", "scale"]}
        params["force_cpu"] = self.device.type == "cpu"
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @abc.abstractmethod
    def load_model(self):
        """Override to perform any model initialization code"""

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Override to perform pre-processing

        Parameters
        ----------
        batch
            For detection plugins, this will be a batch of square, padded, images at model input
            size in the plugin's color order, image format and data range.

            For align plugins this will be a face detection ROI bounding box (batch size, left,
            top, right, bottom) as INT32.

            For all other plugins this will be a batch of aligned face images at model input
            size in the plugin's color order, image format and data range

        Returns
        -------
        For align plugins, this should be an adjustment of the detected face's bounding box to cut
        a square out of the original image for feeding the model. Out of bounds values are allowed,
        as these will be handled. This bounding box will be used to prepare the image at the
        correct size for feeding the model.

        For all other plugins, any pre-processing (eg normalization) should be applied ready for
        feeding the model.
        """
        return batch

    @abc.abstractmethod
    def process(self, batch: np.ndarray) -> np.ndarray:
        """Override to perform processing. This is where the model should be called

        Parameters
        ----------
        batch
            For detection plugins, this will be a batch of square, padded, images at model input
            size in the correct format for feeding the model

            For align, mask and identity plugins this will be a batch of square face patches at
            model input size in the correct format for feeding the model

        Returns
        -------
        This can return any numpy array, but it must be a numpy array. For detect plugins that can
        return several results, usually in a list, then this must be an object array
        """

    def post_process(self, batch: np.ndarray) -> npt.NDArray[np.float32]:
        """Override to perform post-processing

        Parameters
        ----------
        batch
            This will be the output from the previous 'process' step

        Returns
        -------
        For detect plugins this must be an (N, M, left, top, right, bottom) bounding boxes for
        detected faces scaled to model input size as float32. N is the batch size, M is the number
        of detections per batch

        For align plugins this must be an (N, 68, 2) float32 array for each (x, y) landmark point
        for each face in the batch. co-ordinates should be normalized to 0.0 to 1.0 range

        For mask plugins this must be an (N, size, size) float32 image in range 0. - 1.0 for each
        face in the batch

        For identity plugins this must be an (N, M) float32 identity embedding
        """
        return batch

    def _get_device(self, cpu: bool = False) -> torch.device:
        """Get the correctly configured device for running inference

        Parameters
        ----------
        cpu
            ``True`` to force running on the CPU.

        Returns
        -------
        The device that torch should use
        """
        if cpu:
            logger.debug("[%s] CPU mode selected. Returning CPU device context", self.name)
            return torch.device("cpu")

        if torch.cuda.is_available():
            logger.debug("[%s] Cuda available. Returning Cuda device context", self.name)
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            logger.debug("[%s] MPS available. Returning MPS device context", self.name)
            return torch.device("mps")

        logger.debug("[%s] No backends available. Returning CPU device context", self.name)
        return torch.device("cpu")

    def load_torch_model(self, model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
        """Load a PyTorch model, apply the weights and pass a warmup batch through

        This function does not need to be used, but some default Faceswap optimizations are
        performed here, so without using this function you will either need to apply them yourself
        or not have them applied

        Parameters
        ----------
        model
            The Torch model to load
        weights_path
            Full path to the weights file to load

        Returns
        -------
        The loaded model ready for inference
        """
        weights = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(weights)
        model.to(self.device, memory_format=torch.channels_last)  # pyright:ignore[reportCallIssue]
        model.eval()
        if self.compile:
            if not self._compile_warning.is_set():
                self._compile_warning.set()
                logger.info("Compiling PyTorch models...")
            model = torch.compile(model)  # pyright:ignore[reportAssignmentType]

        placeholder = torch.zeros((self.batch_size, 3, self.input_size, self.input_size),
                                  dtype=torch.float32,
                                  device=self.device).to(memory_format=torch.channels_last)
        with torch.inference_mode():
            model(placeholder)
        logger.debug("[%s] Loaded model", self.name)
        return model


class FacePlugin(ExtractPlugin):
    """Base extract plugin that all plugins that work with aligned faces must inherit from.

    Parameters
    ----------
    input_size
        The size of the input required by the plugin. The input will always be square at these
        dimensions
    batch_size
        The batch size that the plugin processes data at. Note: Only the `process` method is
        guaranteed to receive data at this batch size (or less). The other processes may receive
        higher batch sizes for re-processing reasons. Do not rely on this when processing data.
        Default: `1`
    is_rgb
        ``True`` if the plugin expects input images to be RGB rather than BGR. Default: ``False``
    dtype
        A valid datatype that the plugin expects to receive the image at. Default: "float32"
    scale
        The scale that the plugin expects to receive the image at eg: (0, 255) for uint8 images.
        Default: (0, 1)
    force_cpu
        For Torch models, force running on the CPU, rather than the accelerated device. Sets the
        :class:`torch.device` to :attr:`device`. Default: ``False``
    centering
        The centering that the mask should be stored at
    """
    def __init__(self,
                 input_size: int,
                 batch_size: int = 1,
                 is_rgb: bool = False,
                 dtype: str = "float32",
                 scale: tuple[int, int] = (0, 1),
                 force_cpu: bool = False,
                 centering: T.Literal["face", "head", "legacy"] = "face") -> None:
        super().__init__(  # pylint:disable=too-many-arguments,too-many-positional-arguments
            input_size=input_size,
            batch_size=batch_size,
            is_rgb=is_rgb,
            dtype=dtype,
            scale=scale,
            force_cpu=force_cpu)

        self.centering: CenteringType = centering
        """The aligned centering of the image patch to feed the model"""
        self.storage_name = self.__module__.rsplit(".", maxsplit=1)[-1].replace("_", "-")
        """str : Dictionary safe name for storing the serialized data"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        retval = super().__repr__()[:-1]
        return retval + f", centering={repr(self.centering)})"


__all__ = get_module_objects(__name__)

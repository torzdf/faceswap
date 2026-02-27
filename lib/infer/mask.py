#! /usr/env/bin/python3
"""Handles face masking plugins and runners """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np
import numpy.typing as npt

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.extract import extract_config as cfg

from .objects import ExtractorBatchMask, ExtractSignal
from .runner import ExtractRunnerFace

if T.TYPE_CHECKING:
    from .objects import ExtractorBatch


logger = logging.getLogger(__name__)


class Mask(ExtractRunnerFace):
    # pylint:disable=duplicate-code
    """Responsible for running Masking plugins within the extract pipeline

    Parameters
    ----------
    plugin
        The plugin that this runner is to use
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    def __init__(self, plugin: str, config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(plugin, config_file=config_file)

    # Pre-processing
    def _pre_process_aligned(self, batch: ExtractorBatch, matrices: npt.NDArray[np.float32]
                             ) -> npt.NDArray[np.uint8]:
        """Pre-process the data when the input are aligned faces. Sub-crops the feed images from
        the aligned images and adds the ROI mask to the alpha channel

        Parameters
        ----------
        batch
            The inbound batch object containing aligned faces
        matrices
            The adjustment matrices for taking the image patch from the full frame for plugin input

        Returns
        -------
        The prepared images with ROI mask in the alpha channel
        """
        assert batch.frame_sizes is not None, (
            "[Mask] Frame sizes must be provided when input is aligned faces")

        dtype = batch.images[0].dtype
        retval = np.empty((len(batch.bboxes), self._input_size, self._input_size, 4), dtype=dtype)
        retval[..., :3] = self._get_faces_aligned(batch.images,
                                                  batch.frame_ids,
                                                  batch.aligned.offsets_head,
                                                  getattr(batch.aligned,
                                                          self._aligned_offsets_name))

        scales = np.hypot(matrices[..., 0, 0], matrices[..., 1, 0])  # Always same x/y scaling
        interps = np.where(scales > 1.0, cv2.INTER_LINEAR, cv2.INTER_AREA)
        size = (self._input_size, self._input_size)
        for idx, (mat, interp) in enumerate(zip(matrices, interps)):
            mask = np.ones((batch.frame_sizes[batch.frame_ids[idx]]), dtype=dtype) * 255
            retval[idx, :, :, 3] = cv2.warpAffine(mask, mat, size, flags=interp)

        return retval

    def pre_process(self) -> None:
        """Obtain the aligned face images at the requested size, centering and image format.
        Peform any plugin specific pre-processing"""
        process = "pre_process"
        logger.debug("[%s.%s] Starting process", self._plugin.name, process)
        for batch in self._get_data(process):
            self._maybe_log_warning(batch.landmark_type)
            matrices = self._get_matrices(getattr(batch.aligned, self._aligned_mat_name))

            if batch.is_aligned:
                data = self._pre_process_aligned(batch, matrices)
            else:
                data = self._get_faces(batch.images, batch.frame_ids, matrices, with_alpha=True)

            data = self._format_images(data)
            batch.matrices = data[..., -1]  # Hacky re-use of unused matrices property for ROI
            batch.data = self._plugin.pre_process(data[..., :3])
            batch.masks[self.storage_name] = ExtractorBatchMask(self._centering, matrices)
            self._put_data(process, batch)

        logger.debug("[%s.%s] Finished process", self._plugin.name, process)
        self._put_data(process, ExtractSignal.SHUTDOWN)

    # Post-processing
    @classmethod
    def _crop_out_of_bounds(cls, masks: npt.NDArray[np.float32], roi_masks: npt.NDArray[np.float32]
                            ) -> None:
        """Un-mask any area of the predicted mask that falls outside of the original frame.

        Parameters
        ----------
        masks
            The predicted masks from the plugin
        roi_mask
            The roi masks. In frame is white, out of frame is black
        """
        if np.all(roi_masks):
            return  # All of the masks are within the frame
        roi_masks = roi_masks[..., None] if masks.ndim == 4 else roi_masks
        masks *= roi_masks

    def post_process(self) -> None:
        """Perform mask post processing.

        Obtains the final output from the mask plugins and masks any part of the face patch that
        goes out of bounds

        Batch object then sent to the next plugin runner
        """
        process = "post_process"
        logger.debug("[%s.%s] Starting process", self._plugin.name, process)

        storage_size = cfg.mask_storage_size()
        if 0 < storage_size < 64:
            logger.warning("Updating mask storage size from %s to 64", storage_size)
            storage_size = 64

        for batch in self._get_data(process):
            masks = batch.data
            if self._overridden[process]:
                masks = self._plugin.post_process(masks)
            self._crop_out_of_bounds(masks, batch.matrices)

            if storage_size == 0:
                storage_size = masks.shape[1]
                logger.debug("[%s.%s] Updated storage size to %s",
                             self._plugin.name, process, storage_size)

            batch.masks[self.storage_name].masks = (masks * 255.).astype("uint8")
            batch.masks[self.storage_name].storage_size = storage_size
            self._put_data(process, batch)

        logger.debug("[%s.%s] Finished process", self._plugin.name, process)
        self._put_data(process, ExtractSignal.SHUTDOWN)


__all__ = get_module_objects(__name__)

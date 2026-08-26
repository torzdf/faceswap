#! /usr/env/bin/python3
""" Generates real-time training previews and timelapse recordings during model training

This optional module provides preview functionality that generates visual samples of predicted
faces during training sessions, allowing users to monitor convergence in real-time. It supports two
primary use cases: live preview generation during active training (PreviewUnit) and periodic
timelapse recording at save intervals for later analysis (TimelapseUnit). The core Samples class
handles all image composition logic including background patches, foreground predictions, mask
overlays, and header labels for swap/identity identification

The module integrates with the Faceswap training loop system and supports optional mask
visualization with configurable opacity and color. It can handle both RGB and alpha-channel inputs
depending on whether learn_mask is enabled in the configuration
"""
from __future__ import annotations

import logging
import os
import typing as T

import cv2
import numpy as np
import torch

from lib.image import hex_to_rgb
from lib.logger import format_array, parse_class_init
from lib.utils import get_module_objects
from lib.training.units.core import TrainingUnit
from lib.training.data import get_label, PreviewLoader
from plugins.train import train_config as mod_cfg
from plugins.train.trainer import trainer_config as trn_cfg

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.model.plugin import FaceswapModel
    from lib.training.training_loop import TrainingEvents, TrainStep


logger = logging.getLogger(__name__)


class Samples():
    """ Container class for managing preview sample composition and display

    This class handles all aspects of constructing the visual preview images used during training.
    It manages background patches, foreground predictions, mask overlays with configurable opacity
    and color, and generates header labels showing swap/identity relationships.

    Parameters
    ----------
    coverage_ratio
        Coverage ratio of the visual patch that the model is being trained at
    has_mask
        Whether the model is being trained with a mask (either through loss or as a side task)
    mask_opacity
        Opacity percentage for the mask color overlay (0-100)
    mask_color
        Hex string representing the color used for mask overlays (e.g., "red", "#FF0000").
    """
    def __init__(self,
                 coverage_ratio: float,
                 has_mask: bool,
                 mask_opacity: int,
                 mask_color: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._coverage_ratio = coverage_ratio
        self._has_mask = has_mask
        self._mask_opacity = mask_opacity / 100.0
        self._mask_color = mask_color
        self._mask_color_array = (
            np.array(hex_to_rgb(mask_color),
                     dtype=np.float32)[..., 2::-1] / 255.).astype(np.float32)

        self._name = self.__class__.__name__
        self._display_mask = has_mask

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = ", ".join(f"{k[1:]}={v!r}" for k, v in self.__dict__.items()
                           if k in ("_coverage_ratio", "_has_mask", "_mask_opacity",
                                    "_mask_color"))
        return f"{self._name}({params})"

    def _toggle_mask_display(self) -> None:
        """ Toggle visibility of the mask overlay and INFO log the action """
        if not self._has_mask:
            return

        display_mask = not self._display_mask
        print("\x1b[2K", end="\r")  # Clear last line
        logger.info("Toggling mask display %s...", "on" if display_mask else "off")
        self._display_mask = display_mask

    def _get_background(self,
                        targets: npt.NDArray[np.float32],
                        patch_size: int,
                        padding: int) -> npt.NDArray[np.float32]:
        """ Create background patches by repeating target images with swap box corners overlay

        Parameters
        ----------
        targets
            Target images with shape (num_identies, num_previews, H, W, C) in float32 dtype
        patch_size
            The size of individual patches in pixels (side length for square patches)
        padding
            Half-width of the central region where swap box corner overlay is applied

        Returns
        -------
        Background patches with shape (num_swaps, num_swaps + 1, num_previews, H, W, C) in float32
        dtype

        Raises
        ------
        AssertionError
            If coverage_ratio equals 1.0 since background is only needed when blending exists
        """
        num_swaps = targets.shape[0]
        assert self._coverage_ratio != 1.0, "Background only required for coverage != 1.0"
        retval = np.empty((num_swaps, num_swaps + 1, *targets.shape[1:4], 3), dtype=np.float32)
        length = patch_size // 4
        t_l, b_r = (padding - 1, patch_size - padding + 1)
        retval[:] = np.repeat(targets[:, None, ..., :3], 3, axis=1)
        retval[:, :, :, t_l:t_l + length, t_l:t_l + length] = self._mask_color_array
        retval[:, :, :, t_l:t_l + length, b_r - length:b_r] = self._mask_color_array
        retval[:, :, :, b_r - length:b_r, b_r - length:b_r] = self._mask_color_array
        retval[:, :, :, b_r - length:b_r, t_l:t_l + length] = self._mask_color_array
        logger.debug("[%s] Created background display patches: %s",
                     self._name, format_array(retval))
        return retval

    def _get_foreground(self,
                        predictions: npt.NDArray[np.float32],
                        targets: npt.NDArray[np.float32],
                        patch_size: int,
                        padding: int) -> npt.NDArray[np.float32]:
        """ Construct foreground patches from predictions or cropped targets

        Parameters
        ----------
        predictions
            Predicted face images with shape (num_identities, num_identities, num_previews, H, W,
            C) in float32 dtype
        targets
            Target images matching the prediction batch dimensions for reference cropping
        patch_size
            The size of individual patches in pixels (side length for square patches)
        padding
            Half-width defining the central region extraction boundaries

        Returns
        -------
        Foreground patches with shape (num_swaps, num_swaps + 1, num_previews, H, W, C)
        in float32 dtype
        """
        num_swaps = predictions.shape[0]
        retval = np.empty((num_swaps, num_swaps + 1, *predictions.shape[2:5], 3),
                          dtype=np.float32)

        retval[:, 1:] = predictions[..., :3]

        if self._coverage_ratio == 1.:
            retval[:, 0] = targets[..., :3]
        else:
            retval[:, 0] = targets[:,
                                   :,
                                   padding:patch_size - padding,
                                   padding:patch_size - padding,
                                   :3]

        logger.debug("[%s] Created foreground display patches: %s",
                     self._name, format_array(retval))
        return retval

    def _apply_masks(self,
                     patches: npt.NDArray[np.float32],
                     predictions: npt.NDArray[np.float32],
                     targets: npt.NDArray[np.float32],
                     patch_size: int,
                     padding: int) -> npt.NDArray[np.float32]:
        """ Apply mask overlays to patches when display is enabled

        Parameters
        ----------
        patches
            Background/foreground composite patches to apply masks to (num_identities,
            num_identities, num_previews, H, W, C) in float32
        predictions
            Prediction images used for mask alpha extraction when learn_mask is enabled
        targets
            Target images providing original alpha values for mask computation
        patch_size
            The size of individual patches in pixels (side length for square patches)
        padding
            Half-width defining the central region boundaries for mask application

        Returns
        -------
        Masked patches with blend applied, maintaining shape (num_identities, num_identities,
        num_previews, H, W, C) in float32
        """
        if not self._display_mask:
            return patches

        if predictions.shape[-1] == 4:  # Learn mask is enabled
            masks = np.zeros(patches.shape[:-1], dtype=np.float32)
            masks[:, 0] = targets[..., -1]
            pred = predictions[..., -1]

            if self._coverage_ratio == 1.0:
                masks[:, 1:] = pred
            else:
                masks[:, 1:, :, padding:patch_size - padding, padding:patch_size - padding] = pred
        else:
            masks = np.repeat(targets[:, None, ..., -1], 3, axis=1)
        masks = 1. - masks
        overlay = np.ones_like(patches, dtype=np.float32) * self._mask_color_array
        masks *= self._mask_opacity
        overlay *= masks[..., None]
        patches *= (1. - masks[..., None])
        retval = patches + T.cast("npt.NDArray[np.float32]", overlay)
        logger.debug("[%s] Applied masks: %s", self._name, format_array(retval))
        return retval

    def _get_headers(self, num_swaps: int, patch_width: int  # pylint:disable=too-many-locals
                     ) -> npt.NDArray[np.uint8]:
        """ Generate header labels showing swap-to-identity relationships for each preview patch

        Parameters
        ----------
        num_swaps
            Number of swap positions in the preview grid
        patch_width
            Width of individual patches in pixels, used for font size calculation and layout

        Returns
        -------
        Header labels as a single-row image with shape (height, total_width, 3) in uint8 dtype
        """
        labels = [
            get_label(i, num_swaps) + (f" > {get_label(i + j, num_swaps, next_identity=True)}"
                                       if j > 0 else "")
            for i in range(num_swaps)
            for j in range(num_swaps + 1)
        ]
        cols = len(labels)
        height = int(patch_width / 4.5)
        headers = np.zeros((cols, height, patch_width, 3), dtype="uint8") + 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        scaling = patch_width / 140
        text_sizes = [cv2.getTextSize(labels[idx], font, scaling, 1)[0]
                      for idx in range(len(labels))]
        t_y = int((height + text_sizes[0][1]) / 2)
        t_x = [int((patch_width - text_sizes[i][0]) / 2) for i in range(cols)]
        thickness = max(1, patch_width // 64)
        logger.debug("[%s] labels: %s, text_sizes: %s, text_x: %s, text_y: %s, thickness: %s, "
                     "scaling: %s",
                     self._name, labels, text_sizes, t_x, t_y, thickness, scaling)
        for idx, (text, header) in enumerate(zip(labels, headers)):
            cv2.putText(header,
                        text,
                        (t_x[idx], t_y),
                        font,
                        scaling,
                        (0, 0, 0),
                        thickness,
                        lineType=cv2.LINE_AA)
        retval = headers.swapaxes(0, 1).reshape((height, patch_width * cols, 3))
        logger.debug("[%s] Headers: %s", self._name, format_array(retval))
        return retval

    def _create_image(self, patches: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """ Compose headers and patches into the final preview image

        Parameters
        ----------
        patches
            Background/foreground composite patches with shape (num_identities,
            num_identities + 1, num_previews, H, W, C) in float32 dtype that will be transposed and
            reshaped for display

        Returns
        -------
        Final preview image as a single 2D array with shape (height, width, 3) in uint8 dtype
        """
        headers = self._get_headers(patches.shape[0], patches.shape[-2])
        src_side, img_count, identities, rows, cols, channels = patches.shape
        images = (patches.transpose(2, 3, 0, 1, 4, 5).reshape((rows * identities,
                                                               cols * src_side * img_count,
                                                               channels)) * 255.).astype(np.uint8)
        if images.shape[0] > images.shape[1]:
            height = len(images) // 2
            images = np.concatenate([images[:height], images[height:]], axis=1)
            headers = np.concatenate([headers, headers], axis=1)
        retval = np.concatenate([headers, images], axis=0)
        logger.debug("[%s] Created preview: %s", self._name, format_array(retval))
        return retval

    def get_preview(self,
                    predictions: npt.NDArray[np.float32],
                    targets: npt.NDArray[np.float32],
                    toggle_mask: bool,
                    ) -> npt.NDArray[np.uint8]:
        """ Generate a complete preview image from predictions and target images

        Orchestrates all steps to produce the final preview: creates foreground patches from
        predictions (or cropped targets), optionally adds background when blending is needed,
        applies mask overlays if configured and visible, generates headers with identity labels,
        and composes everything into a single display image.

        Parameters
        ----------
        predictions
            Predicted face images with shape (num_identities, num_identities, num_previews, H, W,
            C) in float32 dtype
        targets
            Target images with shape (num_identities, num_previews, H, W, C) for background and
            cropping reference
        toggle_mask
            Whether to toggle mask display visibility.

        Returns
        -------
        Final preview image as a 2D array with shape (height, width, 3) in uint8 dtype, ready for
        display or saving to disk. Contains headers on top followed by grid of prediction images
        below.
        """
        if toggle_mask:
            self._toggle_mask_display()

        patch_size = targets.shape[-2]
        pad = (patch_size - predictions.shape[-2]) // 2

        logger.debug("[%s] Showing sample. Predictions: %s, targets: %s, patch_size: %s, "
                     "padding: %s",
                     self._name, format_array(predictions), format_array(targets),
                     patch_size, pad)

        foreground = self._get_foreground(predictions, targets, patch_size, pad)

        if self._coverage_ratio != 1.0:
            patches = self._get_background(targets, patch_size, pad)
            patches[:, :, :, pad:patch_size - pad, pad:patch_size - pad] = foreground
        else:
            patches = foreground

        patches = self._apply_masks(patches, predictions, targets, patch_size, pad)
        return self._create_image(patches)


class EvaluateUnit(TrainingUnit):
    """ Evaluation unit for generating preview images during training

    This unit evaluates the trained model on live batches to generate visual previews of
    face swapping results in real-time. It manages the model inference pipeline with batched
    processing, handles both RGB and alpha-channel inputs depending on learn_mask configuration,
    and prepares data through a PreviewLoader that feeds images from configured folders

    The unit supports optional mask visualization (learn_mask/mask loss) for monitoring convergence
    of mask prediction alongside face predictions, as well as being able to focus on the swap area.
    It processes input batches in chunks defined by the augmentation batch size setting to handle
    large datasets efficiently without loading everything into memory at once

    Parameters
    ----------
    model
        The training faceswap model plugin instance used for generating predictions during
        evaluation
    """
    def __init__(self, model: FaceswapModel) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._model = model

        self._batch_size = trn_cfg.Augmentation.preview_images()
        self._learn_mask = mod_cfg.Loss.learn_mask()
        self._output_size = model.info.output_size
        self._is_rgb = model.plugin.is_rgb

        self._samples = Samples(mod_cfg.coverage() / 100.,
                                mod_cfg.Loss.learn_mask() or mod_cfg.Loss.penalized_mask_loss(),
                                trn_cfg.Augmentation.mask_opacity(),
                                trn_cfg.Augmentation.mask_color())
        self._loader: PreviewLoader  # set by child
        self._device: torch.Device   # set in on_load

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return f"{self.__class__.__name__}(model={self._model!r})"

    def on_load(self, loop: TrainStep) -> None:
        """ Initialize the evaluation unit and set inference device

        Retrieves the torch.device from the training loop for running model inference. This ensures
        GPU/CPU resources can be properly allocated during ``step``

        Parameters
        ----------
        loop
            The training step object managing this unit's lifecycle
        """
        self._device = loop.device
        logger.debug("[%s] Set device to: '%s'", self.log_name, str(self._device))

    def _get_predictions(self, feed: torch.Tensor) -> npt.NDArray[np.float32]:
        """ Generate prediction arrays for a batch of input images

        Parameters
        ----------
        feed
            Input batch of image tensors with shape (num_identites, num_previews, C, H_in, W_in)

        Returns
        -------
        Predictions array with shape (num_identites, num_previews, H_out, W_out, C) where
        C is 3 for RGB or 4 if learn_mask includes alpha channel. float32 range [0, 1]
        """
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        retval = np.empty((feed.shape[0],
                           feed.shape[1],
                           self._output_size,
                           self._output_size, ndim),
                          dtype=np.float32)
        for idx in range(0, feed.shape[1], self._batch_size):
            feed_batch = feed[:, idx:idx + self._batch_size]
            feed_size = feed_batch.shape[1]
            is_padded = feed_size < self._batch_size

            if is_padded:
                holder = torch.empty((feed_batch.shape[0],
                                      self._batch_size,
                                      *feed_batch.shape[2:]),
                                     dtype=feed.dtype)
                logger.debug("%s Padding undersized batch of shape %s to %s",
                             self.log_name, feed_batch.shape, holder.shape)
                holder[:, :feed_size] = feed_batch
                feed_batch = holder

            with torch.inference_mode():
                out = [y.cpu().numpy().transpose(0, 2, 3, 1)
                       for x in self._model.plugin(list(feed_batch.to(self._device)))
                       for y in x
                       if y.shape[2] == self._output_size]  # Filter multi-scale output
            if mod_cfg.Loss.learn_mask():  # Apply mask to alpha channel
                out = [np.concatenate(out[i:i + 2], axis=-1) for i in range(0, len(out), 2)]
            out_arr = np.stack(out, axis=0)
            if is_padded:
                out_arr = out_arr[:, :feed_size]
            retval[:, idx:idx + feed_size] = out_arr
        return retval

    def _get_samples(self, toggle_mask: bool = False) -> npt.NDArray[np.uint8]:
        """ Generate preview images for all side combinations in current batch

        Parameters
        ----------
        toggle_mask
            Whether to toggle mask display visibility. Default: ``False``

        Returns
        -------
        Final preview image as a 2D array with shape (height, width, 3) in uint8 dtype
        """
        feed, target = next(self._loader)
        num_sides = feed.shape[0]
        ndim = 4 if self._learn_mask else 3
        predictions: npt.NDArray[np.float32] = np.empty((num_sides,
                                                         num_sides,
                                                         target.shape[1],
                                                         self._output_size,
                                                         self._output_size,
                                                         ndim),
                                                        dtype=np.float32)
        logger.debug("[%s] feed: %s, target: %s, predictions_holder: %s",
                     self.log_name, feed.shape, target.shape, predictions.shape)
        for side_idx in range(num_sides):
            rolled_feed = torch.roll(feed, shifts=side_idx, dims=0)
            pred = self._get_predictions(rolled_feed)
            for input_idx in range(num_sides):
                original_idx = (input_idx - side_idx) % num_sides
                predictions[original_idx, side_idx] = pred[input_idx]

        targets = target.cpu().numpy()
        if self._is_rgb:
            predictions[..., :3] = predictions[..., 2::-1]
            targets[..., :3] = targets[..., 2::-1]
        logger.debug("[%s] Got preview images: predictions: %s, targets: %s",
                     self.log_name, format_array(predictions), format_array(targets))

        return self._samples.get_preview(predictions, targets, toggle_mask)


class PreviewUnit(EvaluateUnit):
    """ Live preview unit for real-time training monitoring

    This unit generates and displays live previews during active training sessions by periodically
    evaluating the model on batches from configured input folders. It integrates with the
    TrainingEvents system to handle mask toggling requests and displays generated previews through
    cache files that can be viewed in the GUI or logged for analysis

    The PreviewUnit wraps an EvaluateUnit and adds a PreviewLoader specifically configured with
    RandomSampler for shuffling batches during live preview generation. It automatically triggers
    preview generation on start (via on_update call) and responds to toggle_mask events by
    refreshing previews and clearing the event flag so subsequent requests are processed

    Parameters
    ----------
    model
        The training faceswap model plugin instance used for generating live previews during
        training
    folders
        List of folder paths containing input images to use as sources for preview generation
    """
    def __init__(self, model: FaceswapModel, folders: list[str]) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(model)
        self._loader = PreviewLoader(model.info.input_size,
                                     self._output_size,
                                     "rgb" if self._is_rgb else "bgr",
                                     folders,
                                     self._batch_size,
                                     torch.utils.data.RandomSampler)
        self._events: TrainingEvents  # set in on_load

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        retval = super().__repr__()[:-1]
        return (f"{retval}, "
                f"folders={self._loader.input_folders!r})")

    def on_load(self, loop: TrainStep) -> None:
        """ Initialize the preview unit and trigger first preview generation

        Calls parent's on_load to set up device, then retrieves TrainingEvents reference from the
        training loop. Immediately generates the first preview upon initialization by calling
        on_update(), which will be handled by the main thread

        Parameters
        ----------
        loop
            The training step object managing this unit's lifecycle
        """
        super().on_load(loop)
        self._events = loop.events
        logger.debug("%s Referenced events: %s", self.log_name, loop.events)
        self.on_update()

    def on_start(self) -> None:
        """ Trigger a preview update on the first real training iteration """
        self.on_update()

    def on_update(self) -> None:
        """ Generate a new preview when triggered by training event

        Checks the toggle_mask event flag to determine if mask visibility should change. If
        toggled, clears the flag after processing to ensure single response per request. Then
        generates previews from current batch data and sends them through events.set_preview() for
        handling in the main thread
        """
        logger.debug("%s Generating preview", self.log_name)
        toggle_mask = self._events.toggle_mask.is_set()
        if toggle_mask:
            logger.debug("%s Toggle mask received. Resetting flag", self.log_name)
            self._events.toggle_mask.clear()
        self._events.set_preview(self._get_samples(toggle_mask=toggle_mask))


class TimelapseUnit(EvaluateUnit):
    """ Timelapse recording unit for training progress documentation

    This unit generates preview images at save intervals and records them as an image sequence to
    document model convergence over time. Unlike PreviewUnit which runs continuously during
    training, TimelapseUnit only produces output when on_save() is called (typically every N
    iterations or when user saves). Each saved timelapse image contains previews from the current
    batch and is written as a JPEG file with an 8-digit iteration number for easy chronological
    ordering.

    The unit uses SequentialSampler instead of RandomSampler to ensure consistent, reproducible
    batches at each save point. It creates the output folder automatically if it doesn't exist and
    logs confirmation when timelapse files are successfully written

    Parameters
    ----------
    model
        The training faceswap model plugin instance used for generating timelapse previews during
        training
    input_folders
        List of folder paths containing input images to use as sources for preview generation
    output_folder
        Path to the output folder where timelapse JPEG files will be saved with iteration numbers
    """
    def __init__(self, model: FaceswapModel, input_folders: list[str], output_folder: str) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(model)
        self._output_folder = output_folder
        self._loader = self._get_loader(model.info.input_size, input_folders)

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        retval = super().__repr__()[:-1]
        return (f"{retval}, input_folders={self._loader.input_folders!r}, "
                f"output_folder={self._output_folder!r})")

    def _get_loader(self, input_size: int, input_folders: list[str]) -> PreviewLoader:
        """ Create a SequentialSampler loader for timelapse with available image count

        Parameters
        ----------
        input_size
            The original input image resolution used to configure the PreviewLoader's expected size
        input_folders
            List of folder paths containing source images for preview generation

        Returns
        -------
        Configured loader instance with SequentialSampler and calculated num_samples parameter
        """
        avail_images = min(len([fname for fname in os.listdir(folder)
                                if os.path.splitext(fname)[-1].lower() == ".png"])
                           for folder in input_folders)
        num_samples = min(self._batch_size, avail_images)
        logger.debug("%s preview count: %s, available_images: %s, timelapse count: %s",
                     self.log_name, self._batch_size, avail_images, num_samples)
        retval = PreviewLoader(input_size,
                               self._output_size,
                               "rgb" if self._is_rgb else "bgr",
                               input_folders,
                               self._batch_size,
                               torch.utils.data.SequentialSampler,
                               num_samples=num_samples)
        logger.debug("%s data loader: %s", self.log_name, retval)
        return retval

    def on_save(self, iteration: int) -> None:
        """ Generate timelapse preview and save as JPEG file

        Creates a preview image from the current batch (using SequentialSampler for consistent
        sampling), ensures output folder exists by creating it if needed, then saves the preview
        as a JPEG with an 8-digit zero-padded iteration number

        Parameters
        ----------
        iteration
            Current training iteration number, used both in logger message and filename
        """
        logger.debug("%s Generating timelapse [%s]", self.log_name, iteration)
        samples = self._get_samples()
        if not os.path.exists(self._output_folder):
            logger.debug("%s Creating timelapse output folder: '%s'",
                         self.log_name, self._output_folder)
            os.makedirs(self._output_folder)

        filename = os.path.join(self._output_folder, f"{iteration:08d}.jpg")
        cv2.imwrite(filename, samples)

        logger.debug("%s Created timelapse: '%s'", self.log_name, filename)


__all__ = get_module_objects(__name__)

#! /usr/env/bin/python3
""" Handles the creation of preview images for saving, GUI and display """
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
from lib.training.data import get_label, PreviewLoader
from plugins.train import train_config as mod_cfg
from plugins.train.trainer import trainer_config as trn_cfg

from .core import TrainingUnit

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.model.plugin.handler import FaceswapModel
    from lib.training.training_loop import TrainingEvents, TrainStep


logger = logging.getLogger(__name__)


class Samples():
    """ Compile samples for display for preview and time-lapse

    This class generates composite image previews by combining source patches, model predictions,
    and optional mask overlays into a single display-ready image. It supports configurable coverage
    ratios (full face vs cropped), mask toggling for debugging, and proper header labeling

    Parameters
    ----------
    coverage_ratio
        Ratio of face area to crop from the training image. Set to 1.0 for full face patches,
        or a smaller value (e.g., 0.8) to zoom in more
    has_mask
        ``True`` if the model was trained with mask learning enabled. Controls where mask
        overlays are generated from
    mask_opacity
        The opacity percentage (0-100) for the mask overlay when displayed. Used to visualize
        learned face boundaries without obscuring underlying content
    mask_color
        A hex RGB string specifying the color used for mask overlays

    Notes
    -----
    The Samples object is stateful and maintains display settings like mask visibility. Use
    toggle_mask_display() to show/hide masks interactively during GUI preview sessions.
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
        """ Pretty print for logging """
        params = ", ".join(f"{k[1:]}={v!r}" for k, v in self.__dict__.items()
                           if k in ("_coverage_ratio", "_has_mask", "_mask_opacity",
                                    "_mask_color"))
        return f"{self._name}({params})"

    def _toggle_mask_display(self) -> None:
        """ Toggle the mask overlay on or off depending on user input during preview sessions

        Notes
        -----
        If has_mask is False (model wasn't trained with masks), this method has no effect.
        """
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
        """ Generate the background image patches for preview composition

        Creates base patches filled with source (ground truth) images. For coverage ratios
        less than 100%, creates a mask-colored box around where predictions will be placed,
        allowing visualization of context during training.

        Parameters
        ----------
        targets
            The (BGR) target patches stacked as (src_side, batch_size, height, width, channels)
        patch_size
            The size of each final face patch in pixels
        padding
            The padding around the prediction area to show context

        Returns
        -------
        The background image patches shaped (src_side, num_src + 1, batch_size, height, width, 3)

        Notes
        -----
        For full coverage (ratio = 1.0), only source images are displayed. For partial coverage,
        the mask-colored box helps identify where face predictions will be placed during training.
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
        """ Generate the foreground patches containing model predictions for overlay

        Extracts prediction areas from source images (for context) and places full-size
        face predictions in their designated positions. For full coverage models, only the
        target background is shown without cropping.

        Parameters
        ----------
        predictions
            The (BGR) predictions shaped (src_side, dst_side, batch_size, height, width, channels)
        targets
            The (BGR) target patches shaped (src_side, batch_size, height, width, channels)
        patch_size
            The size of each final face patch in pixels
        padding
            The padding around the prediction area to show context

        Returns
        -------
        The foreground image patches shaped (src_side, num_src + 1, batch_size, height, width, 3)
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
        """ Apply learned masks to preview patches if mask display is enabled

        Handles two cases: when the model learns a separate alpha channel (for masking),
        and when using pre-trained binary masks from targets. Masks are blended with opacity
        settings for visual clarity.

        Parameters
        ----------
        image
            The image patches shaped (src_side, num_src + 1, batch_size, height, width, 3) to have
            masks applied
        predictions
            The (BGR) predictions shaped (src_side, dst_side, batch_size, height, width, channels)
        targets
            The (BGR) targets shaped (src_side, batch_size, height, width, channels)
        patch_size
            The size of each final face patch in pixels
        padding
            The padding around the prediction area to show context

        Returns
        -------
        The masked image patches ready for display
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
        """ Generate header row with identity labels for preview image columns

        Creates readable text headers showing which source (A) maps to which target (B),
        using the model's side labels and proper formatting for multi-swap scenarios.

        Parameters
        ----------
        num_swaps
            The number of swap instances within the model (e.g., 2 for A→B, B→C training)
        patch_width
            The width in pixels of each preview column header

        Returns
        -------
        The column headings array shaped (height, columns * patch_width, 3) as uint8
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
        """ Compose the final preview image with headers and layout

        Arranges all patches into a single display-ready image by transposing dimensions,
        stacking source images side-by-side, and adding header row for column labels.

        Parameters
        ----------
        patches
            The final image patches shaped (src_side, num_src + 1, batch_size, height, width, 3)

        Returns
        -------
        The final preview image as uint8 array ready for display or saving

        Notes
        -----
        Images are arranged horizontally if they're wider than tall. Headers appear above
        the source columns to identify which transformations each column represents.
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
        """ Compile a complete preview image from predictions and target patches

        This is the main entry point for generating display-ready previews. Combines background,
        foreground (predictions), optional masks, and headers into a single image suitable
        for GUI display, saving to disk, or timelapse sequences.

        Parameters
        ----------
        predictions
            The (BGR) predictions shaped (src_side, dst_side, batch_size, height, width, channels)
        targets
            Full size BGR face patches at 100% coverage for patching predictions into in
            (A, B, ...) order
        mask_toggle
            ``True`` if the mask should be toggled from its current state

        Returns
        -------
        A compiled preview image as uint8 array ready for display or saving

        Notes
        -----
        The method handles:

        1. Converting predictions to foreground patches with proper cropping
        2. Creating background patches when using partial coverage ratios
        3. Blending masks onto face regions if enabled and visible
        4. Arranging all components into final display layout
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
    """ Base unit for generating preview images during training and timelapse

    This class provides common functionality for creating preview outputs. It generates
    face prediction visualizations by feeding batches through the model, processing results
    with configured masks/coverage settings, and returning display-ready images.

    Parameters
    ----------
    model
        The FaceswapModel object containing the trained neural network, device info, and
        configuration for preview generation (batch size, mask learning state, RGB mode)

    Notes
    -----
    Derived classes PreviewUnit and TimelapseUnit
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
        self._device: torch.Device   # set in on_start

    def __repr__(self) -> str:
        """ String representation for debugging and logging """
        return f"{self.__class__.__name__}(model={self._model!r})"

    def on_start(self, loop: TrainStep) -> None:
        """ Initialize device context when training commences

        Parameters
        ----------
        loop
            The configured training loop object about to process its first batch

        Notes
        -----
        This method establishes the device reference for inference operations.
        """
        self._device = loop.device
        logger.debug("[%s] Set device to: '%s'", self.log_name, str(self._device))

    def _get_predictions(self, feed: torch.Tensor) -> npt.NDArray[np.float32]:
        """ Obtain preview predictions from the model by chunking into batch-sized feeds

        This method processes input batches through the model in inference mode, handling
        variable-length inputs by padding to batch size. Multi-scale outputs are filtered
        to keep only those matching the configured output resolution.

        Parameters
        ----------
        feed
            The input tensor to obtain predictions from with shape (num_sides, N, height, width)

        Returns
        -------
        The processed predictions as a numpy array of float32 values shaped:
        (num_sides, num_sides, output_size, output_size, ndim) where ndim is 4 if masks are used

        Notes
        -----
        Works in inference_mode for speed. Masks are concatenated to alpha channel when learn_mask
        is enabled.
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
    """ Handles creation of preview images during training sessions

    This unit generates real-time previews for the GUI by feeding batches through the model
    and composing display-ready images with proper headers, masks, and layout. It's called
    at save intervals or through manual trigger to update the live preview viewer.

    Parameters
    ----------
    model
        The FaceswapModel object containing the trained neural network used for generating
        previews during training
    folders
        List of folder paths containing input images/videos for each side of the model

    Notes
    -----
    This unit creates interactive previews that display:

    1. Current model predictions overlaid on target patches
    2. Identity headers showing source→target relationships (e.g., "A > B")
    3. Optional mask overlays if the model was trained using a mask

    The preview is updated every save interval and displayed in the GUI for monitoring
    training progress and visual quality assessment during sessions.
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
        self._events: TrainingEvents  # set in on_start

    def __repr__(self) -> str:
        """ String representation for debugging and logging """
        retval = super().__repr__()[:-1]
        return (f"{retval}, "
                f"folders={self._loader.input_folders!r})")

    def on_start(self, loop: TrainStep) -> None:
        """ Generate the first preview image on model start """
        super().on_start(loop)
        self._events = loop.events
        logger.debug("%s Referenced events: %s", self.log_name, loop.events)
        self.on_update()

    def on_update(self) -> None:
        """ Generate a preview image at the current training iteration

        Fetches the next batch from the data loader, runs inference through the model,
        composes a preview with headers and optional masks, then returns both the image
        and status message for GUI display. This is called during save intervals and by manual
        trigger to update the live preview viewer.

        Parameters
        ----------
        iteration
            The current training iteration number used for logging purposes

        Returns
        -------
        samples
            The composed preview image as uint8 array ready for display
        status_message
            Instructions displayed in the GUI header

        Notes
        -----
        Actions performed in order:

        1. Log debug message
        2. Generate preview via _get_samples() method
        3. Set the preview image to the loop.event object
        """
        logger.debug("%s Generating preview", self.log_name)
        toggle_mask = self._events.toggle_mask.is_set()
        if toggle_mask:
            logger.debug("%s Toggle mask received. Resetting flag", self.log_name)
            self._events.toggle_mask.clear()
        self._events.set_preview(self._get_samples(toggle_mask=toggle_mask))


class TimelapseUnit(EvaluateUnit):
    """ Handles creation of timelapse images for tracking training progress over time

    This unit generates periodic snapshots of the model's visual improvements by processing
    batches through the model and saving complete preview images to a dedicated output folder.
    Useful for creating animation sequences showing how face swapping quality evolves during
    training.

    Parameters
    ----------
    model
        The FaceswapModel object containing the trained neural network used for generating
        timelapse snapshots
    input_folders
        List of folder paths containing input images/videos for each side of the model. Used to
        sample initial frames from these sources when creating preview batches

    output_folder
        Directory path where generated timelapse images will be saved with sequential numbering

    Notes
    -----
    This unit creates time-lapse sequences that:

    1. Sample initial frames from input folders at save intervals
    2. Process through the model to generate face-swapped previews
    3. Save as sequential JPG files (00000000.jpg, 00000001.jpg, etc.)

    The output folder is created automatically if it doesn't exist. Images are saved in
    chronological order allowing easy playback to visualize model improvement over training epochs.
    """
    def __init__(self, model: FaceswapModel, input_folders: list[str], output_folder: str) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(model)
        self._output_folder = output_folder
        self._loader = self._get_loader(model.info.input_size, input_folders)

    def __repr__(self) -> str:
        """ String representation for debugging and logging """
        retval = super().__repr__()[:-1]
        return (f"{retval}, input_folders={self._loader.input_folders!r}, "
                f"output_folder={self._output_folder!r})")

    def _get_loader(self, input_size: int, input_folders: list[str]) -> PreviewLoader:
        """ Create and configure the data loader for timelapse image generation

        This method sets up a sequential sampler to ensure consistent sampling from each input
        folder, preventing random sampling that could produce inconsistent timelapse sequences.

        Parameters
        ----------
        input_size
            The resolution size of input images (height and width in pixels)
        input_folders
            List of folder paths containing source images/videos for each model side

        Returns
        -------
        A configured PreviewLoader instance ready for sampling timelapse batches

        Notes
        -----
        Uses SequentialSampler instead of RandomSampler to ensure consistent image ordering across
        different training sessions. The number of samples is limited by the smaller batch size or
        available images in the input folders, whichever is fewer.
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
        """ Generate and save a timelapse preview image at the current training iteration

        Processes batches through the model to create visual snapshots of face-swapping quality.
        Images are saved sequentially in the output folder with zero-padded filenames for easy
        chronological playback. The output folder is created automatically if needed.

        Parameters
        ----------
        iteration
            The current training iteration number. Used as the filename prefix for the generated
            timelapse image

        Notes
        -----
        Actions performed in order:

        1. Log debug message with current iteration count
        2. Generate preview via _get_samples() method (runs model inference)
        3. Create output folder if it doesn't exist
        4. Save image as JPG with zero-padded filename (08-digit format)
        5. Log confirmation of saved timelapse file

        The iteration number ensures each save gets a unique filename, preventing overwrites
        and creating a complete chronological record of training progress.

        Notes
        -----
        Unlike PreviewUnit which displays images in GUI, this unit saves to disk for later
        analysis. The SequentialSampler ensures consistent sampling from input folders across
        different runs.
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

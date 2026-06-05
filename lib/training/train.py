#! /usr/env/bin/python3
"""Run the training loop for a training plugin"""
from __future__ import annotations

import logging
import os
import typing as T
import time

import cv2
import numpy as np

import torch

from lib.logger import format_array, parse_class_init
from lib.model.plugin.model_info import Info
from lib.model.plugin import TrainHandler, TrainConfigure

from lib.torch_utils import get_device
from lib.training.preview import Samples
from lib.training.data import get_label, PreviewLoader, TrainLoader
from lib.training.tensorboard import TorchTensorBoard
from lib.utils import get_module_objects, FaceswapError
from plugins.train import train_config as mod_cfg

from plugins.train.trainer import trainer_config as trn_cfg
from plugins.train.trainer.base import TrainConfig

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from collections.abc import Callable
    from plugins.train.model.base import ModelPlugin
    from plugins.train.trainer.base import TrainerBase
    from .loss import BatchLoss

logger = logging.getLogger(__name__)


class LossHandler:
    """Handles the logging and output of loss values from the model

    Parameters
    ----------
    device
        The device that is being used for training
    nan_protection
        ``True`` to enable NaN protection
    """
    def __init__(self, device: torch.Device, nan_protection: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._device = device
        self._nan_protection = nan_protection

        self._tensorboard: TorchTensorBoard | None = None
        self._averages: dict[T.Literal["unweighted", "weighted"], dict[str, torch.Tensor]] = {}
        self._loss_count = 0

    def __repr__(self) -> str:
        """Pretty print for logging"""
        return (f"{self.__class__.__name__}(device={repr(self._device)}, "
                f"nan_protection={repr(self._nan_protection)})")

    def set_tensorboard(self,
                        model: ModelPlugin,
                        model_folder: str,
                        model_name: str,
                        session_id: int) -> None:
        """Set up Tensorboard callback for logging loss.

        Parameters
        ----------
        model
            The faceswap model that is to be trained
        model_folder
            The full path to the folder that the model is saved to
        model_name
            The name of the model plugin
        session_id
            The ID of the session about to commence training
        """
        logger.debug("[LossHandler] Setting up TensorBoard Logging. model: %s, model_folder: %s, "
                     "model_name: %s, session_id: %s",
                     model, repr(model_folder), repr(model_name), session_id)
        log_dir = os.path.join(model_folder, f"{model_name}_logs", f"session_{session_id}")
        tensorboard = TorchTensorBoard(log_dir=log_dir,
                                       write_graph=True,
                                       update_freq="batch")
        tensorboard.set_model(model)
        logger.verbose("Enabled TensorBoard Logging")  # type: ignore[attr-defined]
        self._tensorboard = tensorboard

    def _reset_averages(self, names: list[str] | None = None) -> None:
        """Reset the loss averages to zero

        Parameters
        ----------
        names
            The name of the loss functions to track when initially setting up
        """
        names = list(self._averages["unweighted"]) if names is None else names
        self._averages = {w: {k: torch.zeros((1, ), dtype=torch.float32, device=self._device)
                              for k in names}
                          for w in ("unweighted", "weighted")}
        self._loss_count = 0
        logger.debug("[LossHandler] Reset loss averages: %s", self._averages)

    def _handle_nan(self, loss: list[BatchLoss]) -> None:
        """Handle NaNs detected in loss

        Raises
        ------
        FaceswapError
            If a NaN is detected, a :class:`FaceswapError` will be raised
        """
        if not self._nan_protection:
            return
        if all(torch.isfinite(val.total).all() for val in loss):
            return

        loss_str = ", ".join(f"Loss {get_label(i, len(loss))}: {round(x.total.item(), 6)}"
                             for i, x in enumerate(loss))
        msg = f"NaN Detected. {loss_str}"
        failed = ", ".join(f"{key}({get_label(i, len(loss))})"
                           for i, out in enumerate(loss)
                           for unweighted in out.unweighted
                           for key, sub_loss in unweighted.items()
                           if not torch.isfinite(sub_loss).all())
        if failed:
            msg += f". The loss function(s) that NaN'd: {failed}"
        logger.critical(msg)
        raise FaceswapError("A NaN was detected and you have NaN protection enabled. Training "
                            "has been terminated.")

    def _log_tensorboard(self, loss: list[BatchLoss], iteration: int) -> None:
        """Log current loss to Tensorboard log files

        Parameters
        ----------
        loss
            The detached loss scalars for the batch on the training device in order (A, B, ...)
        iteration
            The total training step being processed
        """
        if not self._tensorboard:
            return
        logger.trace("[LossHandler] Updating TensorBoard. iteration: %s log: %s",  # type: ignore
                     iteration, loss)
        logs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "total": T.cast(torch.Tensor, sum(x.total for x in loss))}
        for i, out in enumerate(loss):
            lbl = get_label(i, len(loss))
            for idx, (w, u) in enumerate(zip(out.weighted, out.unweighted)):
                key = lbl if len(out.unweighted) == 1 else f"{lbl}_{idx}"
                weighted = {k: v.mean() for k, v in w.items()}
                unweighted = {k: v.mean() for k, v in u.items()}
                logs[f"face_{key}"] = T.cast(torch.Tensor, sum(weighted.values()))
                logs[f"weighted_{key}"] = weighted
                logs[f"unweighted_{key}"] = unweighted
            if out.mask is not None:
                logs[f"mask_{lbl}"] = out.mask.mean()
        self._tensorboard.on_train_batch_end(iteration, logs=logs)

    def _update_averages(self, loss: list[BatchLoss]) -> None:
        """Store the total running weighted and unweighted averages for each loss function since
        the last save iteration.

        Parameters
        ----------
        loss
            The list of detached loss outputs on the training device in order (A, B, ...)
        """
        contrib = [x.get_contributions() for x in loss]
        totals = {w: {k: sum(d[w][k] for d in contrib) for k in contrib[0][w]} for w in contrib[0]}
        if not self._averages:
            self._reset_averages(names=list(totals["unweighted"]))
        self._loss_count += 1
        self._averages = {a: {k: self._averages[a][k] + (totals[a][k] -
                                                         self._averages[a][k]) / self._loss_count
                              for k in self._averages[a]}
                          for a in self._averages}

    def _print_loss(self, loss: list[BatchLoss], iteration: int) -> None:
        """Outputs the loss for the current iteration to the console.

        Parameters
        ----------
        loss
            The detached loss output from the model on the training device in order (A, B, ...
        iteration
            The total training step being processed
        """
        totals = {i: x.total.item() for i, x in enumerate(loss)}
        output = ", ".join(f"Loss {get_label(k, len(totals))}: {v:.5f}"
                           for k, v in totals.items())
        timestamp = time.strftime("%H:%M:%S")
        output = f"[{timestamp}] [#{iteration:05d}] {output}"
        print(f"{output}", end="\r")

    def step(self, loss: list[BatchLoss], iteration: int) -> None:
        """Handle the logging and output of loss values for a batch

        Parameters
        ----------
        loss
            The collated attached loss values on the training device in order (A, B, ...)
        iteration
            The current total training iteration
        """
        loss = [x.detach() for x in loss]
        self._handle_nan(loss)
        self._log_tensorboard(loss, iteration)
        self._update_averages(loss)
        self._print_loss(loss, iteration)

    def _output_contributions(self) -> None:
        """Output the contributions for each loss function since the last save"""
        totals = {w: sum(m.values()) for w, m in self._averages.items()}
        ratios = {w: {k: round(((v / totals[w]) * 100.).item(), 1) for k, v in m.items()}
                  for w, m in self._averages.items()}
        msg = "Ratios since save [Weighted (Unweighted)]: "
        msg += ", ".join(f"{k}: {ratios['weighted'][k]}% ({ratios['unweighted'][k]}%)"
                         for k in ratios["unweighted"])
        logger.info(msg)

    def on_save(self) -> float:
        """Logging actions to perform when the model is saved

        Returns
        -------
        The average total loss since the last save iteration
        """
        assert self._tensorboard is not None  # TODO why does this work? Test with no-logs
        self._tensorboard.on_save()
        self._output_contributions()

        retval = T.cast(torch.Tensor, sum(self._averages["weighted"].values())).item()
        self._reset_averages()

        logger.debug("[LossHandler] Average total since last save: %s", retval)
        return retval

    def close(self) -> None:
        """Stop Tensorboard logging.

        Tensorboard logging needs to be explicitly shutdown on training termination.
        """
        if not self._tensorboard:
            return
        logger.debug("[LossHandler] Ending Tensorboard Session: %s", self._tensorboard)
        self._tensorboard.on_train_end()


class Trainer:  # pylint:disable=too-many-instance-attributes
    """Handles the feeding of training images to Faceswap models, the generation of Tensorboard
    logs and the creation of sample/time-lapse preview images.

    Parameters
    ----------
    trainer_name
        The name of the trainer plugin to load
    model_name
        The name of the faceswap model plugin that is to be trained
    train_config:
        The Training Configuration options
    preview
        ``True`` to generate previews
    timelapse_folders
        The input folders to create timelapse images from. Default: ``None`` (no timelapse)
    timelapse_output
        The folder to output timelapse images. Default: "" (no timelapse)
    summary
        ``True`` to just output a summary of the model and exit. ``False`` to train.
        Default: ``False``
    config_file
        The custom location to load configuration options from or ``None`` if default location
    """
    def __init__(self,
                 trainer_name: str,
                 model_name: str,
                 train_config: TrainConfig,
                 preview: bool,
                 timelapse_folders: list[str] | None = None,
                 timelapse_output: str = "",
                 summary: bool = False,
                 config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        mod_cfg.load_config(config_file=config_file)

        self._train_config = train_config
        self._model_handler = TrainHandler(model_name,
                                           len(train_config.folders),
                                           train_config.model_folder)
        self._model_info = Info(self._model_handler.model)
        self._model_info.summary(logger.info if summary
                                 else logger.verbose)  # type:ignore[attr-defined]
        if summary:
            return

        self._device = get_device()
        self._trainer = self._configure_model(trainer_name,
                                              train_config.warmup_steps,
                                              train_config.batch_size)
        self._train_loader = self._get_train_loader()

        self._exit_early = self._handle_lr_finder()
        if self._exit_early:
            logger.debug("[Trainer] Exiting from LR Finder")
            return

        self._tester = Tester(self._trainer,
                              self._model_info,
                              self._device,
                              self._train_config.folders if preview else None,
                              timelapse_folders,
                              timelapse_output)
        self._loss_handler = LossHandler(self._device, mod_cfg.nan_protection())
        if not train_config.no_logs:
            self._loss_handler.set_tensorboard(self._model_handler.model,
                                               train_config.model_folder,
                                               model_name,
                                               self._model_handler.session_id + 1)

    @property
    def exit_early(self) -> bool:
        """``True`` if the trainer should exit early, without performing any training steps"""
        return self._exit_early

    def _configure_model(self,
                         trainer_name: str,
                         warmup_steps: int,
                         batch_size: int) -> TrainerBase:
        """Add the model and the loss functions to the training plug in and move to the correct
        device

        Parameters
        ----------
        trainer_name
            The name of the trainer plugin to load

        Returns
        -------
        The Faceswap trainer plugin with the Faceswap model loaded
        """
        is_new = not self._model_handler.model_exists
        train_config = TrainConfigure(self._model_info,
                                      loss_config=mod_cfg.Loss,
                                      optimizer_config=mod_cfg.Optimizer,
                                      icnr_init=mod_cfg.icnr_init() and is_new,
                                      conv_aware_init=mod_cfg.conv_aware_init() and is_new,
                                      mixed_precision=mod_cfg.mixed_precision(),
                                      reflect_padding=mod_cfg.reflect_padding())
        if train_config.mixed_precision:
            logger.info("Enabled Auto Mixed Precision")
        retval = self._model_handler.configure_model(trainer_name=trainer_name,
                                                     train_config=train_config,
                                                     warmup_steps=warmup_steps,
                                                     batch_size=batch_size)
        logger.debug("[Trainer] Configured model and trainer: %s", retval)
        return retval

    def _get_train_loader(self) -> TrainLoader:
        """Get the loaders for training the model

        Returns
        -------
        The loaders for feeding the model's training loop
        """
        out_sizes = [[x[1] for x in side if x[0] != 1] for side in self._model_info.output_shapes]
        num_sides = len(self._train_config.folders)
        assert len(out_sizes) % num_sides == 0, (
            f"Output count ({len(out_sizes)}) doesn't match number of inputs ({num_sides})")

        assert len(set(x for side in out_sizes
                       for x in side)) == len(out_sizes[0]), "Sizes for each output must match"
        retval = TrainLoader(self._model_info.input_size,
                             tuple(out_sizes[0]),
                             "rgb" if self._model_handler.model.is_rgb else "bgr",
                             self._train_config,
                             self._trainer.sampler)
        logger.debug("[Trainer] data loader: %s", retval)
        return retval

    def _handle_lr_finder(self) -> bool:  # TODO
        """Handle the learning rate finder.

        If this is a new model, then find the optimal learning rate and return ``True`` if user has
        just requested the graph, otherwise return ``False`` to continue training

        If it as existing model, set the learning rate to the value found by the learning rate
        finder and return ``False`` to continue training

        Returns
        -------
        ``True`` if the learning rate finder options dictate that training should not continue
        after finding the optimal leaning rate
        """
        if not self._train_config.lr_finder:
            return False

        assert self._model_handler.optimizer is not None
        if self._model_handler.state.lr_finder > -1:
            learning_rate = self._model_handler.state.lr_finder
            logger.info("Setting learning rate from Learning Rate Finder to %s",
                        f"{learning_rate:.1e}")
            self._model_handler.optimizer.set_lr(learning_rate)
            self._model_handler.state.update_session_config("learning_rate", learning_rate)
            return False

        if self._model_handler.total_iterations == 0 and self._model_handler.session_id == 1:
            success = self._model_handler.optimizer.find_learning_rate(
                self,
                mod_cfg.lr_finder_iterations(),
                1e-10,
                1e-1,
                T.cast(T.Literal["default", "aggressive", "extreme"],
                       mod_cfg.lr_finder_strength()),
                T.cast(T.Literal["set", "graph_and_set", "graph_and_exit"],
                       mod_cfg.lr_finder_mode())
                )
            return mod_cfg.lr_finder_mode() == "graph_and_exit" or not success

        logger.debug("[Trainer] No learning rate finder rate. Not setting")
        return False

    def toggle_mask(self) -> None:
        """Toggle the mask overlay on or off based on user input."""
        self._tester.toggle_mask()

    def train_one_step(self,
                       viewer: Callable[[np.ndarray, str], None] | None,
                       do_timelapse: bool = False) -> None:
        """Running training on a batch of images for each side.

        Triggered from the training cycle in :class:`scripts.train.Train`.

        * Runs a training batch through the model.

        * Outputs the iteration's loss values to the console

        * Logs loss to Tensorboard, if logging is requested.

        * If a preview or time-lapse has been requested, then pushes sample images through the \
        model to generate the previews

        * Creates a snapshot if the total iterations trained so far meet the requested snapshot \
        criteria

        Notes
        -----
        As every iteration is called explicitly, the Parameters defined should always be ``None``
        except on save iterations.

        Parameters
        ----------
        viewer
            The function that will display the preview image
        do_timelapse
            ``True`` to generate a timelapse preview image
        """
        iteration = self._model_handler.total_iterations + 1
        logger.trace("[Trainer] Training one step: (iteration: %s)",  # type:ignore[attr-defined]
                     iteration)
        do_snapshot = (self._train_config.snapshot_interval != 0 and
                       iteration - 1 >= self._train_config.snapshot_interval and
                       (iteration - 1) % self._train_config.snapshot_interval == 0)

        inputs, targets, meta = next(self._train_loader)
        loss = self._trainer.step([i.to(self._device) for i in inputs],
                                  [t.to(self._device) for t in targets],
                                  meta.to(self._device),
                                  self._model_handler.loss,
                                  self._model_handler.optimizer)
        self._loss_handler.step(loss, iteration)
        self._model_handler.step(self._trainer.batch_size)

        if do_snapshot:
            pass  # TODO
            # self._model.io.snapshot()
        if viewer is not None:
            self._tester(viewer, do_timelapse)

    def save(self, is_exit: bool = False) -> None:
        """Save the model

        Parameters
        ----------
        is_exit
            ``True`` if save has been called on model exit. Default: ``False``
        """
        save_optimizer = mod_cfg.Optimizer.save_optimizer()
        save_optimizer = save_optimizer == "always" or (is_exit and save_optimizer == "exit")
        average_loss = self._loss_handler.on_save()
        self._model_handler.save(average_loss, save_optimizer)
        if is_exit:
            self._loss_handler.close()


class Tester:
    """Responsible for running tests through the model for previews/timelapses

    Parameters
    ----------
    trainer
        The faceswap trainer plugin to obtain previews from
    model_info
        The object containing information about the loaded model
    device
        The device that the model resides on
    preview_folders
        List of folders, for each side, to load training images for preview from. ``None`` if
        preview disabled
    timelapse_folders
        The input folders to create timelapse images from. Default: ``None`` (no timelapse)
    timelapse_output
        The folder to output timelapse images. Default: "" (no timelapse)
    """
    def __init__(self,
                 trainer: TrainerBase,
                 model_info: Info,
                 device: torch.Device,
                 preview_folders: list[str] | None,
                 timelapse_folders: list[str] | None = None,
                 timelapse_output: str = "") -> None:
        logger.debug(parse_class_init(locals()))
        self._trainer = trainer
        self._model_info = model_info
        self._device = device
        self._timelapse_folders = [] if timelapse_folders is None else timelapse_folders
        self._timelapse_output = timelapse_output
        self._batch_size = trn_cfg.Augmentation.preview_images()
        self._samples = Samples(mod_cfg.coverage() / 100.,
                                mod_cfg.Loss.learn_mask() or mod_cfg.Loss.penalized_mask_loss(),
                                trn_cfg.Augmentation.mask_opacity(),
                                trn_cfg.Augmentation.mask_color())

        self._preview_loader = self._get_preview_loader(preview_folders)
        self._timelapse_loader = self._get_timelapse_loader()

    def _get_preview_loader(self, preview_folders: list[str] | None) -> PreviewLoader | None:
        """Get the loader for generating previews whilst training the model

        Parameters
        ----------
        preview_folders
            list of folders to read images from for each side being trained

        Returns
        -------
        The loader for generating preview images during training or ``None`` if previews are
        disabled
        """
        if not preview_folders:
            return None
        retval = PreviewLoader(self._model_info.input_size,
                               self._model_info.output_size,
                               "rgb" if self._trainer.model.is_rgb else "bgr",
                               preview_folders,
                               self._batch_size,
                               torch.utils.data.RandomSampler)
        logger.debug("[Trainer] Preview data loader: %s", retval)
        return retval

    def _get_timelapse_loader(self) -> PreviewLoader | None:
        """Get the loader for generating timelapse images whilst training the model

        Returns
        -------
        The loaders for timelapse preview images during training or ``None`` if previews are
        disabled
        """
        if not self._timelapse_folders or not self._timelapse_output:
            return None
        num_images = trn_cfg.Augmentation.preview_images()
        avail_images = min(len([fname for fname in os.listdir(folder)
                                if os.path.splitext(fname)[-1].lower() == ".png"])
                           for folder in self._timelapse_folders)
        num_samples = min(num_images, avail_images)
        logger.debug("[Train] preview count: %s, available_images: %s, timelapse count: %s",
                     num_images, avail_images, num_samples)
        retval = PreviewLoader(self._model_info.input_size,
                               self._model_info.output_size,
                               "rgb" if self._trainer.model.is_rgb else "bgr",
                               self._timelapse_folders,
                               self._batch_size,
                               torch.utils.data.SequentialSampler,
                               num_samples=num_samples)
        logger.debug("[Trainer] Preview data loader: %s", retval)
        return retval

    def _get_predictions(self, feed: torch.Tensor) -> npt.NDArray[np.float32]:
        """Obtain preview predictions from the model, chunking feeds into the model's batch size

        Parameters
        ----------
        feed
            The input tensor to obtain predictions from the model in shape (num_sides, N, height,
            width, 3)

        Returns
        -------
        The predictions from the model for the given preview feed
        """
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        retval = np.empty((feed.shape[0],
                           feed.shape[1],
                           self._model_info.output_size,
                           self._model_info.output_size, ndim),
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
                logger.debug("[Trainer] Padding undersized batch of shape %s to %s",
                             feed_batch.shape, holder.shape)
                holder[:, :feed_size] = feed_batch
                feed_batch = holder

            with torch.inference_mode():
                out = [y.cpu().numpy().transpose(0, 2, 3, 1)
                       for x in self._trainer.model(list(feed_batch.to(self._device)))
                       for y in x
                       if y.shape[2] == self._model_info.output_size]  # Filter multi-scale output
            if mod_cfg.Loss.learn_mask():  # Apply mask to alpha channel
                out = [np.concatenate(out[i:i + 2], axis=-1) for i in range(0, len(out), 2)]
            out_arr = np.stack(out, axis=0)
            if is_padded:
                out_arr = out_arr[:, :feed_size]
            retval[:, idx:idx + feed_size] = out_arr
        return retval

    def __call__(self,  # pylint:disable=too-many-locals
                 viewer: Callable[[np.ndarray, str], None],
                 do_timelapse: bool) -> None:
        """Update the preview viewer and timelapse output

        Parameters
        ----------
        viewer
            The function that will display the preview image
        do_timelapse
            ``True`` to generate a timelapse preview image
        """
        if self._preview_loader is None and not do_timelapse:
            return

        if do_timelapse:
            assert self._timelapse_loader is not None
            loader = self._timelapse_loader
        else:
            assert self._preview_loader is not None
            loader = self._preview_loader
        feed, target = next(loader)

        num_sides = feed.shape[0]
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        predictions: npt.NDArray[np.float32] = np.empty((num_sides,
                                                         num_sides,
                                                         target.shape[1],
                                                         self._model_info.output_size,
                                                         self._model_info.output_size,
                                                         ndim),
                                                        dtype=np.float32)
        logger.debug("[Trainer] feed: %s, target: %s, predictions_holder: %s",
                     feed.shape, target.shape, predictions.shape)
        for side_idx in range(num_sides):
            rolled_feed = torch.roll(feed, shifts=side_idx, dims=0)
            pred = self._get_predictions(rolled_feed)
            for input_idx in range(num_sides):
                original_idx = (input_idx - side_idx) % num_sides
                predictions[original_idx, side_idx] = pred[input_idx]

        targets = target.cpu().numpy()
        if self._trainer.model.is_rgb:
            predictions[..., :3] = predictions[..., 2::-1]
            targets[..., :3] = targets[..., 2::-1]
        logger.debug("[Trainer] Got preview images: predictions: %s, targets: %s",
                     format_array(predictions), format_array(targets))

        samples = self._samples.get_preview(predictions, targets)

        if do_timelapse:
            filename = os.path.join(self._timelapse_output, str(int(time.time())) + ".jpg")
            cv2.imwrite(filename, samples)
            logger.debug("[Trainer] Created time-lapse: '%s'", filename)
            return

        if viewer is not None:
            viewer(samples,
                   "Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. 'F': "
                   "Toggle Screen Fit-Actual Size. 'ENTER': Save and Quit")

    def toggle_mask(self) -> None:
        """Toggle the preview mask display on or off"""
        self._samples.toggle_mask_display()


__all__ = get_module_objects(__name__)

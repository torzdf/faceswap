#! /usr/env/bin/python3
"""Run the training loop for a training plugin"""
from __future__ import annotations

import logging
import os
import typing as T
import time
from dataclasses import dataclass

import cv2
import numpy as np

import torch

from lib.logger import format_array, parse_class_init
from lib.model.plugin.model_info import Info

from lib.torch_utils import get_device
from lib.training.tensorboard import TorchTensorBoard
from lib.utils import get_module_objects, FaceswapError
from plugins.train import train_config as mod_cfg

from plugins.train.trainer import trainer_config as trn_cfg

from .data import AugmentOptions, get_label, PreviewLoader, TrainLoader
from .loss import LossCollator
from .lr_finder import LearningRateFinder
from .optimizer import Optimizer
from .preview import Samples


if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.model.plugin import TrainHandler
    from plugins.train.model.base import ModelPlugin
    from plugins.train.trainer.base import TrainerBase
    from .loss import BatchLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainerReturn:
    """Return object from training loop to calling script"""
    exit: bool = False
    """``True`` to exist training"""
    preview_image: npt.NDArray[np.uint8] | None = None
    """Generated preview image if one should be shown"""
    preview_title: str = ""
    """Title for the generated preview image if one should be shown"""


class LossHandler:
    """Handles the logging and output of loss values from the model

    Parameters
    ----------
    device
        The device that is being used for training
    nan_protection
        ``True`` to enable NaN protection
    input_shapes
        The [(C, H, W)] shape of each input to the model
    model
        The faceswap model that is to be trained for Tensorboard. ``None`` for not logged.
        Default: ``None``
    model_folder
        The full path to the folder that the model is saved to for Tensorboard. ``None`` for not
        logged. Default: ``None``
    model_name
        The name of the model plugin for Tensorboard. ``None`` for not logged. Default: ``None``
    session_id
        The ID of the session about to commence training for Tensorboard. ``None`` for not logged.
        Default: ``None``
    """
    def __init__(self,
                 device: torch.Device,
                 nan_protection: bool,
                 input_shapes: list[tuple[int, int, int]],
                 model: ModelPlugin | None = None,
                 model_folder: str | None = None,
                 model_name: str | None = None,
                 session_id: int | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._device = device
        self._nan_protection = nan_protection
        self._input_shapes = input_shapes
        self._model = model
        self._model_folder = model_folder
        self._model_name = model_name
        self._session_id = session_id

        self._tensorboard: TorchTensorBoard | None = None
        self._averages: dict[T.Literal["unweighted", "weighted"], dict[str, torch.Tensor]] = {}
        self._loss_count = 0

    def _set_tensorboard(self) -> None:
        """Set up Tensorboard callback for logging loss."""
        if self._model is None:
            return
        assert (self._model_folder is not None and
                self._model_name is not None and
                self._session_id is not None)
        logger.debug("[LossHandler] Setting up TensorBoard Logging")
        log_dir = os.path.join(self._model_folder,
                               f"{self._model_name}_logs",
                               f"session_{self._session_id}")
        tensorboard = TorchTensorBoard(log_dir=log_dir,
                                       write_graph=True,
                                       update_freq="batch")
        if self._session_id == 1:
            tensorboard.write_torch_graph(self._model, self._device, self._input_shapes)

        del self._model  # De-ref model as no longer required
        self._model = None
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
        if iteration == 1:
            self._set_tensorboard()

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
        self._tensorboard.step(iteration, logs=logs)

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
        """Handle the logging and output of loss values for a batch.

        - If lrf is not enabled:
            - Detaches all loss values
            - Handles NaNs
            - Logs out to Tensorboard
            - Tracks average since last save
            - Prints the loss to logger if lrf is not enabled

        Parameters
        ----------
        loss
            The collated attached loss values on the training device in order (A, B, ...)
        iteration
            The current total training iteration. Is -1 if training has not fully started (eg
            during learning rate warmup))
        """
        if iteration < 0:
            return
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
        The average total loss since the last save iteration. 0.0 if no loss has been collated (eg
        during learning rate finder)
        """
        if not self._averages:
            logger.debug("[LossHandler] No averages to output. Returning zero loss")
            return 0.0

        if self._tensorboard is not None:
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
    data_folders
        The folders that contain the training images for each input
    model_handler
        The object that handles configuring and loading/saving the Faceswap model during training
    augment_opts
        The training data augmentation options
    warmup_steps
        The number of steps to warmup the learning rate
    no_logs
        ``True`` to disable tensorboard logging
    preview
        ``True`` to generate previews
    timelapse_folders
        The input folders to create timelapse images from. Default: ``None`` (no timelapse)
    timelapse_output
        The folder to output timelapse images. Default: "" (no timelapse)
    summary
        ``True`` to just output a summary of the model and exit. ``False`` to train.
        Default: ``False``
    lr_finder
        ``True`` to use the learning rate finder. Default: ``False``
    config_file
        The custom location to load configuration options from or ``None`` if default location
    """
    def __init__(self,
                 trainer_name: str,
                 data_folders: list[str],
                 model_handler: TrainHandler,
                 augment_opts: AugmentOptions,
                 warmup_steps: int,
                 no_logs: bool,
                 preview: bool,
                 timelapse_folders: list[str] | None = None,
                 timelapse_output: str = "",  # TODO remove and always save to model dir
                 summary: bool = False,
                 lr_finder: bool = False,
                 config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        mod_cfg.load_config(config_file=config_file)  # Set global config

        self._model_handler = model_handler
        self._optimizer = None if summary else Optimizer(self._model_handler.model,
                                                         mod_cfg.Optimizer,
                                                         mod_cfg.mixed_precision(),
                                                         warmup_steps)

        self._model_handler.load_state_dict(self._optimizer)  # Load saved model config
        model_info = Info(self._model_handler.model)
        model_info.summary(logger.info if summary else logger.verbose)  # type:ignore[attr-defined]
        if summary:
            return

        self._device = get_device()
        self._loss_fn, self._trainer = self._configure_model(trainer_name, model_info)
        self._train_loader = self._get_train_loader(model_info.input_size,
                                                    model_info.output_shapes,
                                                    self._model_handler.batch_size,
                                                    data_folders,
                                                    augment_opts)
        self._tester = Tester(trainer_plugin=self._trainer,
                              input_size=model_info.input_size,
                              output_size=model_info.output_size,
                              device=self._device,
                              preview_folders=data_folders if preview else None,
                              timelapse_folders=timelapse_folders,
                              timelapse_output=timelapse_output)

        self._lr_finder = LearningRateFinder(
            enabled=lr_finder,
            model_handler=self._model_handler,
            steps=mod_cfg.lr_finder_iterations(),
            strength=T.cast(T.Literal["default", "aggressive", "extreme"],
                            mod_cfg.lr_finder_strength()),
            mode=T.cast(T.Literal["set", "graph_and_set", "graph_and_exit"],
                        mod_cfg.lr_finder_mode())
        )
        self._loss_handler = LossHandler(self._device,
                                         mod_cfg.nan_protection(),
                                         model_info.input_shapes,
                                         None if no_logs else self._model_handler.model,
                                         None if no_logs else self._model_handler.model_folder,
                                         None if no_logs else self._model_handler.name,
                                         None if no_logs else self._model_handler.session_id + 1)

    @property
    def optimizer(self) -> Optimizer:
        """The currently training optimizer"""
        assert self._optimizer is not None
        return self._optimizer

    def _get_train_loader(self,
                          input_size: int,
                          output_shapes: list[list[tuple[int, int, int]]],
                          batch_size: int,
                          folders: list[str],
                          augment_opts: AugmentOptions) -> TrainLoader:
        """Get the loaders for training the model

        Parameters
        ----------
        input_size
            The input size of the model
        output_shapes
            The shape of each output from the model ([sideA[outputs], sideB[outputs], ...])
        batch_size
            The batch size to load data at
        folders
            The folders to load data from
        augment_opts
            The augmentation options collected from the command line

        Returns
        -------
        The loaders for feeding the model's training loop
        """
        out_sizes = [[x[1] for x in side if x[0] != 1] for side in output_shapes]
        num_sides = len(folders)
        assert len(out_sizes) % num_sides == 0, (
            f"Output count ({len(out_sizes)}) doesn't match number of inputs ({num_sides})")

        assert len(set(x for side in out_sizes
                       for x in side)) == len(out_sizes[0]), "Sizes for each output must match"
        retval = TrainLoader(folders,
                             batch_size,
                             input_size,
                             tuple(out_sizes[0]),
                             "rgb" if self._model_handler.model.is_rgb else "bgr",
                             augment_opts,
                             self._trainer.sampler)
        logger.debug("[Trainer] data loader: %s", retval)
        return retval

    def _configure_model(self, trainer_name: str, model_info: Info
                         ) -> tuple[LossCollator, TrainerBase]:
        """Add the model and the loss functions to the training plugin and move all objects to the
        correct device

        Parameters
        ----------
        trainer_name
            The name of the trainer plugin to load
        model_info
            The object that contains structural information about the model

        Returns
        -------
        loss_fn
            The collated loss functions for training the model
        trainer
            The Faceswap trainer plugin with the Faceswap model loaded
        """
        loss = LossCollator(
            functions=[mod_cfg.Loss.loss_function(),
                       mod_cfg.Loss.loss_function_2(),
                       mod_cfg.Loss.loss_function_3(),
                       mod_cfg.Loss.loss_function_4()],
            weights=[1.0,
                     mod_cfg.Loss.loss_weight_2() / 100.,
                     mod_cfg.Loss.loss_weight_3() / 100.,
                     mod_cfg.Loss.loss_weight_4() / 100.],
            color_order="rgb" if self._model_handler.model.is_rgb else "bgr",
            use_mask=mod_cfg.Loss.penalized_mask_loss(),
            eye_multiplier=mod_cfg.Loss.eye_multiplier(),
            mouth_multiplier=mod_cfg.Loss.mouth_multiplier(),
            smallest_output=min(x[1] for x in model_info.output_shapes[0] if x[0] != 1),
            mask_loss=(None if not mod_cfg.Loss.learn_mask()
                       else mod_cfg.Loss.mask_loss_function()))

        trainer = self._model_handler.configure_model(trainer_name=trainer_name,
                                                      model_info=model_info,
                                                      mixed_precision=mod_cfg.mixed_precision(),
                                                      icnr_init=mod_cfg.icnr_init(),
                                                      conv_aware_init=mod_cfg.conv_aware_init(),
                                                      reflect_padding=mod_cfg.reflect_padding(),
                                                      device=self._device)
        loss.to(self._device)
        self.optimizer.to(self._device)
        logger.debug("[Trainer] Configured model and trainer. loss: %s trainer: %s", loss, trainer)
        return loss, trainer

    def toggle_mask(self) -> None:
        """Toggle the mask overlay on or off based on user input."""
        self._tester.toggle_mask()

    def step(self, gen_preview: bool, timelapse_enabled: bool) -> TrainerReturn:
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
        gen_preview
            ``True`` to force run inference to generate preview images
        timelapse_enabled
            ``True`` if timelapse generation is enabled

        Returns
        -------
        The return object with any relevant information to the caller
        """
        retval = TrainerReturn()
        iteration = -1 if self._lr_finder.is_enabled else self._model_handler.total_iterations + 1
        logger.trace("[Trainer] Training one step: (iteration: %s)",  # type:ignore[attr-defined]
                     iteration)

        inputs, targets, meta = next(self._train_loader)
        loss = self._trainer.step([i.to(self._device) for i in inputs],
                                  [t.to(self._device) for t in targets],
                                  meta.to(self._device),
                                  self._loss_fn,
                                  self.optimizer)
        self._loss_handler.step(loss, iteration)
        self.optimizer.step()

        if self._lr_finder.is_enabled:
            if self._lr_finder.step(T.cast(torch.Tensor, sum(x.total for x in loss))):
                retval.exit = True
                return retval
            if not self._lr_finder.is_enabled:
                logger.debug("[Trainer] LRF Finished")
                return retval
        update_preview = self._model_handler.step(self._loss_handler, self._lr_finder.is_enabled)

        if update_preview and timelapse_enabled:  # TODO no TL on LRF
            self._tester(True)
        if update_preview or gen_preview:
            out = self._tester(False)
            assert out is not None and len(out) == 2
            retval.preview_image = out[0]
            retval.preview_title = out[1]

        return retval

    def save(self, is_exit: bool = False) -> None:
        """Save the model

        Parameters
        ----------
        is_exit
            ``True`` if save has been called on model exit. Default: ``False``
        """
        self._model_handler.save(self._loss_handler, is_exit=is_exit)
        if is_exit:
            self._loss_handler.close()


class Tester:
    """Responsible for running tests through the model for previews/timelapses

    Parameters
    ----------
    trainer_plugin
        The faceswap trainer plugin to obtain previews from
    input_size
        The input size of the model
    output_size
        The output size of the model
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
                 trainer_plugin: TrainerBase,
                 input_size: int,
                 output_size: int,
                 device: torch.Device,
                 preview_folders: list[str] | None,
                 timelapse_folders: list[str] | None = None,
                 timelapse_output: str = "") -> None:
        logger.debug(parse_class_init(locals()))
        self._trainer = trainer_plugin
        self._input_size = input_size
        self._output_size = output_size
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
        retval = PreviewLoader(self._input_size,
                               self._output_size,
                               "rgb" if self._trainer.model.is_rgb else "bgr",
                               preview_folders,
                               self._batch_size,
                               torch.utils.data.RandomSampler)
        logger.debug("[Tester] Preview data loader: %s", retval)
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
        retval = PreviewLoader(self._input_size,
                               self._output_size,
                               "rgb" if self._trainer.model.is_rgb else "bgr",
                               self._timelapse_folders,
                               self._batch_size,
                               torch.utils.data.SequentialSampler,
                               num_samples=num_samples)
        logger.debug("[Tester] Preview data loader: %s", retval)
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
                logger.debug("[Tester] Padding undersized batch of shape %s to %s",
                             feed_batch.shape, holder.shape)
                holder[:, :feed_size] = feed_batch
                feed_batch = holder

            with torch.inference_mode():
                out = [y.cpu().numpy().transpose(0, 2, 3, 1)
                       for x in self._trainer.model(list(feed_batch.to(self._device)))
                       for y in x
                       if y.shape[2] == self._output_size]  # Filter multi-scale output
            if mod_cfg.Loss.learn_mask():  # Apply mask to alpha channel
                out = [np.concatenate(out[i:i + 2], axis=-1) for i in range(0, len(out), 2)]
            out_arr = np.stack(out, axis=0)
            if is_padded:
                out_arr = out_arr[:, :feed_size]
            retval[:, idx:idx + feed_size] = out_arr
        return retval

    def __call__(self,  # pylint:disable=too-many-locals
                 do_timelapse: bool) -> tuple[npt.NDArray[np.uint8], str] | None:
        """Update the preview viewer and timelapse output

        Parameters
        ----------
        do_timelapse
            ``True`` to generate a timelapse preview image, ``False`` to return a preview image

        Returns
        -------
        image
            The composed preview image
        name
            The name (header) of the image

        or ``None`` if a preview has not been generated
        """
        if self._preview_loader is None and not do_timelapse:
            return None

        if do_timelapse:
            logger.debug("[Tester] Generating timelapse")
            assert self._timelapse_loader is not None
            loader = self._timelapse_loader
        else:
            logger.debug("[Tester] Generating preview")
            assert self._preview_loader is not None
            loader = self._preview_loader
        feed, target = next(loader)

        num_sides = feed.shape[0]
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        predictions: npt.NDArray[np.float32] = np.empty((num_sides,
                                                         num_sides,
                                                         target.shape[1],
                                                         self._output_size,
                                                         self._output_size,
                                                         ndim),
                                                        dtype=np.float32)
        logger.debug("[Tester] feed: %s, target: %s, predictions_holder: %s",
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
        logger.debug("[Tester] Got preview images: predictions: %s, targets: %s",
                     format_array(predictions), format_array(targets))

        samples = self._samples.get_preview(predictions, targets)

        if do_timelapse:
            filename = os.path.join(self._timelapse_output, str(int(time.time())) + ".jpg")
            cv2.imwrite(filename, samples)
            logger.debug("[Tester] Created time-lapse: '%s'", filename)
            return None

        return (samples,
                "Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. 'F': "
                "Toggle Screen Fit-Actual Size. 'ENTER': Save and Quit")

    def toggle_mask(self) -> None:
        """Toggle the preview mask display on or off"""
        self._samples.toggle_mask_display()


__all__ = get_module_objects(__name__)

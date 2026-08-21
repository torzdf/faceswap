#!/usr/bin python3
"""Main entry point to the training process of FaceSwap """
from __future__ import annotations
import logging
import os
import sys
import typing as T

from time import sleep
from threading import Event

import cv2
import numpy as np
from torch.utils.data import RandomSampler, DistributedSampler

from lib.gui.utils.image import TRAINING_PREVIEW
from lib.image import validate_faceswap_image
from lib.keypress import KBHit
from lib.logger import parse_class_init
from lib.multithreading import FSThread
from lib.utils import PROJECT_ROOT

from lib.training import Preview, PreviewBuffer, TriggerType
from lib.training.data import TrainLoader
from lib.training.units import (LRFinderUnit, PreviewUnit, TensorBoardUnit, TimelapseUnit,
                                WarmupUnit)
from lib.model.plugin.handler import FaceswapModel
from lib.training.training_loop import TrainingLoop
from lib.utils import (FaceswapError, get_module_objects, handle_deprecated_cli_opts)

from plugins.plugin_loader import PluginLoader

if T.TYPE_CHECKING:
    import argparse
    import numpy.typing as npt
    from lib.training.training_loop import TrainingEvents
    from plugins.train.model.base import ModelPlugin
    from plugins.train.trainer.base import TrainerPlugin

logger = logging.getLogger(__name__)


class PreviewInterface():
    def __init__(self,
                 show_preview: bool,
                 write_preview: bool,
                 gui_preview: bool,
                 training_events: TrainingEvents) -> None:
        logger.debug(parse_class_init(locals()))

        self._events = training_events
        self._triggers: TriggerType = {"toggle_mask": training_events.toggle_mask,
                                       "refresh": training_events.update,
                                       "save": training_events.save,
                                       "quit": training_events.exit,
                                       "shutdown": Event()}
        self._actions = self._get_actions(show_preview, write_preview, gui_preview)
        self._display_buffer = PreviewBuffer() if show_preview else None
        self._display_thread = self._launch_thread() if show_preview else None

    def _get_actions(self, show_preview: bool, write_preview: bool, gui_preview: bool
                     ) -> list[T.Callable]:
        retval = []
        if write_preview:
            retval.append(self._write_preview)
        if gui_preview:
            retval.append(self._gui_preview)
        if show_preview:
            retval.append(self._show_preview)
        logger.debug("[PreviewInterface] Collated Preview actions: %s", retval)
        return retval

    def _launch_thread(self) -> FSThread | None:
        """ Launch Preview window inside thread """
        thread = FSThread(target=Preview,
                          name="preview",
                          args=(self._display_buffer, ),
                          kwargs={"triggers": self._triggers})
        thread.start()
        return thread

    def shutdown(self) -> None:
        """ Shutdown the Preview interface """
        if self._display_thread is None:
            return
        logger.debug("[PreviewInterface] Sending shutdown to preview viewer")
        self._triggers["shutdown"].set()

    @classmethod
    def _write_preview(cls, image: npt.NDArray[np.uint8]) -> None:
        logger.debug("[PreviewInterface] Saving preview to disk")
        img = "training_preview.png"
        img_file = os.path.join(PROJECT_ROOT, img)
        cv2.imwrite(img_file, image)
        logger.debug("[PreviewInterface] Saved preview to: '%s'", img)

    @classmethod
    def _gui_preview(cls, image: npt.NDArray[np.uint8]) -> None:
        logger.debug("[PreviewInterface] Generating preview for GUI")
        img = TRAINING_PREVIEW
        img_file = os.path.join(PROJECT_ROOT, "lib", "gui", ".cache", "preview", img)
        cv2.imwrite(img_file, image)  # pylint:disable=no-member
        logger.debug("[PreviewInterface] Generated preview for GUI: '%s'", img_file)

    def _show_preview(self, image: npt.NDArray[np.uint8]) -> None:
        assert self._display_buffer is not None
        preview_text = ("Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. 'F': "
                        "Toggle Screen Fit-Actual Size. 'ENTER': Save and Quit")
        logger.debug("[PreviewInterface] Generating preview for display: '%s'", preview_text)
        self._display_buffer.add_image(preview_text, image)

    def __call__(self) -> None:
        preview = self._events.get_preview()
        if preview is None:
            return

        logger.debug("[PreviewInterface] Updating preview")
        try:
            for action in self._actions:
                action(preview)
        except Exception as err:
            logging.error("could not preview sample: %s", str(err))
            raise err


class Train():
    """The Faceswap Training Process.

    The training process is responsible for training a model on a set of source faces and a set of
    destination faces.

    The training process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments
        The arguments to be passed to the training process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: argparse.Namespace) -> None:
        logger.debug(parse_class_init(locals()))

        args = handle_deprecated_cli_opts(arguments)
        model = self._output_summary(args)

        self._gui_triggers = self._get_gui_triggers()
        self._training_loop = self._get_training_loop(model, args)
        self._preview = PreviewInterface(args.preview,
                                         args.write_image,
                                         args.redirect_gui,
                                         self._training_loop.events)
        self._args = args
        self._save_now: bool = False

    # BUILD MODEL + SUMMARY OUTPUT
    @classmethod
    def _output_summary(cls, args: argparse.Namespace) -> FaceswapModel:
        """ Output or log a summary of the model. Exit if model summary option was selected """
        model = FaceswapModel(name=args.trainer,
                              model_dir=args.model_dir,
                              num_identities=len([args.input_a, args.input_b]),
                              load_optimizer=not args.summary,
                              config_file=args.config_file)
        model.info.summary(None if args.summary else logger.verbose)  # type:ignore[attr-defined]
        if args.summary:
            sys.exit(0)

        return model

    # TRAINER SETUP
    @classmethod
    def _get_gui_triggers(cls) -> dict[T.Literal["mask", "refresh"], str]:
        gui_cache = os.path.join(PROJECT_ROOT, "lib", "gui", ".cache")
        return {"mask": os.path.join(gui_cache, ".preview_mask_toggle"),
                "refresh": os.path.join(gui_cache, ".preview_trigger")}

    @classmethod
    def _get_trainer(cls, model: ModelPlugin, batch_size: int, distributed: bool) -> TrainerPlugin:
        trainer = "distributed" if distributed else "original"
        if trainer == "distributed":
            import torch  # pylint:disable=import-outside-toplevel
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                logger.warning("Distributed selected but fewer than 2 GPUs detected. Switching "
                               "to Original")
                trainer = "original"
        return PluginLoader.get_trainer(trainer)(model, batch_size)

    @classmethod
    def _get_loader(cls,  # pylint:disable=too-many-positional-arguments,too-many-arguments
                    input_folders: list[str],
                    input_size: int,
                    batch_size: int,
                    output_shapes: list[list[tuple[int, int, int]]],
                    is_rgb: bool,
                    augment_color: bool,
                    flip: bool,
                    warp: bool,
                    warp_to_landmarks: bool,
                    sampler: type[RandomSampler] | type[DistributedSampler]) -> TrainLoader:
        out_sizes = [[x[1] for x in side if x[0] != 1] for side in output_shapes]
        num_sides = len(input_folders)

        assert len(out_sizes) % num_sides == 0, (
            f"Output count ({len(out_sizes)}) doesn't match number of inputs ({num_sides})")
        assert len(set(x for side in out_sizes
                       for x in side)) == len(out_sizes[0]), "Sizes for each output must match"

        retval = TrainLoader(folders=input_folders,
                             batch_size=batch_size,
                             input_size=input_size,
                             output_sizes=tuple(out_sizes[0]),
                             color_order="rgb" if is_rgb else "bgr",
                             augment_color=augment_color,
                             flip=flip,
                             warp=warp,
                             cache_landmarks=warp_to_landmarks,
                             sampler=sampler)
        logger.debug("[Train] data loader: %s", retval)
        return retval

    @classmethod
    def _validate_timelapse(cls, args: argparse.Namespace) -> bool:
        """ Validate timelapse settings

        Returns
        -------
        ``True`` if timelapse is enabled and valid otherwise ``False``
        """
        if not args.timelapse_input_a and not args.timelapse_input_b:
            return False
        if not args.timelapse_input_a or not args.timelapse_input_b:
            raise FaceswapError("To enable the timelapse, you have to supply both the parameters "
                                "--timelapse-input-A and --timelapse-input-B.")

        timelapse_folders: list[str] = [args.timelapse_input_a, args.timelapse_input_b]

        for idx, folder in enumerate(timelapse_folders):
            side = "a" if idx == 0 else "b"
            if folder is not None and not os.path.isdir(folder):
                raise FaceswapError(f"The Timelapse path '{folder}' does not exist")

            training_folder = getattr(args, f"input_{side}")
            if folder == training_folder:
                continue  # Time-lapse folder is training folder

            filenames = [os.path.join(folder, fname) for fname in os.listdir(folder)
                         if os.path.splitext(fname)[-1].lower() == ".png"]
            if not filenames:
                raise FaceswapError(f"The Timelapse path '{folder}' does not contain any valid "
                                    "images")

            if not validate_faceswap_image(filenames[0]):
                logger.error("The input folder '%s' contains images that are not extracted faces.",
                             folder)
                logger.error("You can only train a model on faces generated from Faceswap's "
                             "extract process. Please check your sources and try again.")
                sys.exit(1)
        logger.debug("[Train] Timelapse enabled")
        return True

    def _configure_loop(self,
                        loop: TrainingLoop,
                        model: FaceswapModel,
                        images: list[str],
                        args: argparse.Namespace) -> None:
        if args.use_lr_finder:
            loop.add_unit(LRFinderUnit(start_lr=1e-9, end_lr=1e-2))

        if args.warmup > 0:
            loop.add_unit(WarmupUnit(args.warmup))

        if not args.no_logs:
            loop.add_unit(TensorBoardUnit(args.model_dir,
                                          model.name,
                                          model.state.session_id))

        if args.preview or args.write_image or args.redirect_gui:
            loop.add_unit(PreviewUnit(model, images))

        if self._validate_timelapse(args):
            in_ = [args.timelapse_input_a, args.timelapse_input_b]
            out = os.path.join(args.model_dir, f"{model.name}_timelapse")
            loop.add_unit(TimelapseUnit(model, in_, out))

    def _get_training_loop(self, model: FaceswapModel, args: argparse.Namespace) -> TrainingLoop:
        """ Load the trainer requested for training.

        Returns
        -------
        The model training loop with the requested trainer plugin loaded for the requested model
        """
        logger.debug("[Train] Loading TrainingLoop")

        images = [args.input_a, args.input_b]

        trainer = self._get_trainer(model.plugin, args.batch_size, args.distributed)
        loader = self._get_loader(input_folders=images,
                                  input_size=model.info.input_size,
                                  batch_size=args.batch_size,
                                  output_shapes=model.info.output_shapes,
                                  is_rgb=model.plugin.is_rgb,
                                  augment_color=not args.no_augment_color,
                                  flip=not args.no_flip,
                                  warp=not args.no_warp,
                                  warp_to_landmarks=args.warp_to_landmarks,
                                  sampler=trainer.sampler)
        loop = TrainingLoop(args.iterations,
                            faceswap_model=model,
                            trainer=trainer,
                            loader=loader,
                            save_interval=args.save_interval,
                            snapshot_interval=args.snapshot_interval)
        self._configure_loop(loop, model, images, args)
        logger.debug("[Train] Loaded TrainingLoop %s", loop)
        return loop

    # ## MAIN LOOP ##
    def _output_startup_info(self) -> None:
        """Print the startup information to the console."""
        logger.debug("[Train] Launching Monitor")
        logger.info("===================================================")
        logger.info("  Starting")
        if self._args.preview:
            logger.info("  Using live preview")
        if sys.stdout.isatty():
            logger.info("  Press '%s' to save and quit",
                        "Stop" if self._args.redirect_gui else "ENTER")
        if not self._args.redirect_gui and sys.stdout.isatty():
            logger.info("  Press 'S' to save model weights immediately")
        logger.info("===================================================")

    def _process_gui_triggers(self, events: TrainingEvents) -> None:
        """Check whether a file drop has occurred from the GUI to manually update the preview. """
        if not self._args.redirect_gui:
            return

        gui_events = {"mask": events.toggle_mask, "refresh": events.update}
        for trigger, filename in self._gui_triggers.items():
            if os.path.isfile(filename):
                logger.debug("[Train] GUI Trigger received for: '%s'", trigger)
                gui_events[trigger].set()
                logger.debug("[Train] Removing gui trigger file: %s", filename)
                os.remove(filename)
                if trigger == "refresh":
                    print("\x1b[2K", end="\r")  # Clear last line
                    logger.info("Refresh preview requested...")

    def _check_keypress(self, keypress: KBHit, training_events: TrainingEvents) -> None:
        """ Process any keypresses to training loop events

        Parameters
        ----------
        keypress
            The keypress monitor
        """
        try:
            if keypress.kbhit():
                console_key = keypress.getch()

                if console_key in ("\n", "\r"):
                    logger.debug("[Train] Exit requested")
                    training_events.exit.set()

                if console_key in ("s", "S"):
                    logger.info("Save requested")
                    training_events.save.set()

        except ValueError as err:
            if "I/O operation on closed file" in str(err):
                logger.critical("[Train] Error encountered: %s", str(err))
                training_events.exit.set()
            else:
                raise

    def _shutdown(self, keypress: KBHit):
        logger.debug("[Train] Shutting down")
        self._preview.shutdown()
        self._training_loop.events.exit.set()
        keypress.set_normal_term()

    def _monitor(self) -> None:
        """ Monitor main thread for keypresses/GUI updates and training thread for errors. """
        self._output_startup_info()
        keypress = KBHit(is_gui=self._args.redirect_gui)
        events = self._training_loop.events
        exit_msg = [("Exit requested! The trainer will complete its current cycle, save the model "
                     "and quit."),
                    "This can take a couple of minutes depending on your training speed."]
        if not self._args.redirect_gui:
            exit_msg.append("If you want to kill it now, press Ctrl + c")

        while True:
            try:
                self._training_loop.check_and_re_raise_error()
                self._preview()
                self._process_gui_triggers(events)
                self._check_keypress(keypress, events)

                if self._training_loop.events.exit.is_set():
                    for msg in exit_msg:
                        logger.info(msg)
                    break

                sleep(1)
            except KeyboardInterrupt:
                logger.debug("[Train] Keyboard Interrupt received")
                break
            except Exception as err:
                logger.error("[Train] Train Error caught: %s", str(err))
                self._shutdown(keypress)
                raise

        self._shutdown(keypress)
        logger.debug("[Train] Closed Monitor")

    def process(self) -> None:
        """The entry point for triggering the Training Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.debug("[Train] Starting Training Process")
        self._training_loop.start()
        self._monitor()
        self._training_loop.join()
        sys.stdout.flush()
        logger.debug("[Train] Completed Training Process")


__all__ = get_module_objects(__name__)

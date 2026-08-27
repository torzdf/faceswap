#! /usr/env/bin/python3
"""Main training script entry point that orchestrates model training including preview display and
GUI interaction

This module serves as the command-line interface for running Faceswap training sessions. It
handles:
    - Model loading from checkpoints or fresh initialization via FaceswapModel
    - Data loader setup with augmentation, flipping, warping options
    - Trainer instantiation (original/distributed) based on hardware configuration
    - Training loop construction with configurable save intervals and snapshot frequencies
    - Preview interface management for real-time visualization during training

The script uses the :class:`lib.training.training_loop.TrainingLoop` class to run training in a
background thread while providing interactive controls via keyboard input or GUI triggers (mask
toggle, refresh preview)
"""
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
from lib.training.units import (FreezeWeightsUnit, LoadWeightsUnit, LRFinderUnit, PreviewUnit,
                                TensorBoardUnit, TimelapseUnit, WarmupUnit)
from lib.model.plugin import FaceswapModel
from lib.training import TrainingLoop
from lib.utils import (FaceswapError, get_module_objects, handle_deprecated_cli_opts)

from plugins.plugin_loader import PluginLoader

if T.TYPE_CHECKING:
    import argparse
    import numpy.typing as npt
    from lib.training import TrainingEvents
    from plugins.train.model.base import ModelPlugin
    from plugins.train.trainer.base import TrainerPlugin

logger = logging.getLogger(__name__)


class PreviewInterface():
    """ Manages real-time preview display during training via multiple output modes

    PreviewInterface orchestrates how training previews are displayed to users through three
    possible mechanisms that can be combined based on CLI flags:
        - write_preview : Saves PNG images to disk (training_preview.png) for later inspection
        - gui_preview   : Writes previews to cache directory accessed by GUI for display
        - show_preview  : Launches FSThread displaying previews in window with interactive controls

    Thread Safety:
    ----------
    The PreviewUnit generates previews during training iterations and sets them via TrainingEvents.
    This class reads from events.get_preview() which is thread-safe (protected by lock in
    TrainingEvents). Console preview runs in separate FSThread to avoid blocking main training loop
    with display operations

    Parameters
    ----------
    show_preview
        Whether to launch console-based preview viewer with keyboard controls for interactive
        training monitoring
    write_preview
        Whether to save each preview as PNG file at fixed disk path for later inspection
    gui_preview
        Whether to write preview images to GUI cache directory so they can be displayed in the GUI
    training_events
        Event system object providing thread-safe access to previews generated during training
        iterations

    Notes
    -----
    The PreviewUnit generates previews during training iterations and stores them in
    TrainingEvents. We retrieve and clear the preview atomically via get_preview(), then pass to
    each action. Console preview allows interactive controls ('S' save now, 'R' refresh, 'M' toggle
    mask, etc.) for monitoring
    """
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
        """ Build list of action functions to invoke for each preview image

        Parameters
        ----------
        show_preview
            Whether to include console display action that adds images to PreviewBuffer for viewing
        write_preview
            Whether to include disk write action that saves PNG at fixed path for later inspection
        gui_preview
            Whether to include GUI cache write action that stores image for GUI display

        Returns
        -------
        Filtered list of action function references based on enabled modes
        """
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
        """ Launch background thread for console preview display

        Returns
        -------
        Reference to newly started background thread, or ``None`` if preview is disabled
        """
        thread = FSThread(target=Preview,
                          name="preview",
                          args=(self._display_buffer, ),
                          kwargs={"triggers": self._triggers})
        thread.start()
        return thread

    def shutdown(self) -> None:
        """ Signal preview thread to terminate gracefully by setting shutdown trigger event

        Called during training loop cleanup when all iterations complete or exit is requested.
        Checks if display_thread exists before attempting shutdown - prevents AttributeError on
        PreviewInterface instances where show_preview=False was initially set

        Notes
        -----
        The shutdown trigger in signals the background Preview thread to read and terminate cleanly
        without force-killing process. This allows proper cleanup of memory and file handles
        associated with preview operations
        """
        if self._display_thread is None:
            return
        logger.debug("[PreviewInterface] Sending shutdown to preview viewer")
        self._triggers["shutdown"].set()

    @classmethod
    def _write_preview(cls, image: npt.NDArray[np.uint8]) -> None:
        """ Save preview image as PNG file at fixed disk path for later inspection

        Parameters
        ----------
        image
            Preview image tensor of shape (height, width, 3) in RGB format ready for saving
        """
        logger.debug("[PreviewInterface] Saving preview to disk")
        img = "training_preview.png"
        img_file = os.path.join(PROJECT_ROOT, img)
        cv2.imwrite(img_file, image)
        logger.debug("[PreviewInterface] Saved preview to: '%s'", img)

    @classmethod
    def _gui_preview(cls, image: npt.NDArray[np.uint8]) -> None:
        """ Write preview to GUI cache directory for display in main training window

        Parameters
        ----------
        image
            Preview image tensor of shape (height, width, 3) ready for the GUI
        """
        logger.debug("[PreviewInterface] Generating preview for GUI")
        img = TRAINING_PREVIEW
        img_file = os.path.join(PROJECT_ROOT, "lib", "gui", ".cache", "preview", img)
        cv2.imwrite(img_file, image)  # pylint:disable=no-member
        logger.debug("[PreviewInterface] Generated preview for GUI: '%s'", img_file)

    def _show_preview(self, image: npt.NDArray[np.uint8]) -> None:
        """ Add preview image with text header to console display buffer for interactive viewing

        Parameters
        ----------
        image
            Preview image tensor of shape (height, width, 3) in RGB format ready for display
        """
        assert self._display_buffer is not None
        preview_text = ("Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. 'F': "
                        "Toggle Screen Fit-Actual Size. 'ENTER': Save and Quit")
        logger.debug("[PreviewInterface] Generating preview for display: '%s'", preview_text)
        self._display_buffer.add_image(preview_text, image)

    def __call__(self) -> None:
        """ Main entry point invoked each iteration - retrieve preview and call all registered
        actions

        Called by the monitoring loop (main thread) after each iteration completes. Retrieves
        latest preview from TrainingEvents.get_preview() which atomically reads and clears the
        preview buffer ready for the next generated preview. If no preview available, returns early
        without performing any operations. Otherwise iterates through all registered actions and
        calls each with preview image as argument

        Notes
        -----
        This method is called by the monitoring loop in the main thread
        """
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
    """ Main training orchestrator that builds model, configures loop, and runs training session

    Train is the entry point class for running Faceswap model training from command line arguments.
    It handles all initialization steps including model loading, data loader setup, trainer
    instantiation (original/distributed), optional units registration (LRFinder, Warmup,
    TensorBoard, Preview, Timelapse), and interactive monitoring loop with keyboard controls or GUI
    file triggers

    The class wraps TrainingLoop functionality within a monitoring process that provides real-time
    feedback via console preview (optional) and allows users to interact during training by
    pressing keys ('S' save now, 'ENTER' quit, etc.) when running without GUI redirect mode enabled

    Execution Flow:
    ---------------
    1. __init__()  : Parse arguments, build model summary output if requested, create training
    loop and preview interface
    2. process()   : Start background training thread, enter monitoring loop with periodic checks
    and user input handling
    3. Cleanup     : Join background thread when training completes or exits early due to
    interrupt/quit request

    Parameters
    ----------
    arguments
        Parsed command-line arguments containing all CLI options (model name, batch size,
        iterations, preview settings, etc.)

    Notes
    -----
    The _output_summary method allows inspecting model architecture without running full training -
    useful for verifying loaded configuration before committing to long training sessions (can take
    hours/days depending on batch size and iteration count)

    Distributed trainer requires at least 2 GPUs otherwise falls back to original trainer
    automatically with warning logged. This prevents silent failures when user specifies
    distributed mode but only single GPU available in system configuration
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
        """ Load model and optionally display summary information and exit immediately

        Parameters
        ----------
        args
            Parsed command-line arguments

        Returns
        -------
        Loaded or created model instance ready for training (or exits early if --summary requested)
        """
        model = FaceswapModel(name=args.trainer,
                              model_dir=args.model_dir,
                              num_identities=len([args.input_a, args.input_b]),
                              load_extra_state=not args.summary,
                              config_file=args.config_file)
        model.info.summary(None if args.summary else logger.verbose)  # type:ignore[attr-defined]
        if args.summary:
            sys.exit(0)

        return model

    # TRAINER SETUP
    @classmethod
    def _get_gui_triggers(cls) -> dict[T.Literal["mask", "refresh"], str]:
        """ Build dictionary of GUI trigger file paths for mask toggle and preview refresh

        Returns
        -------
        Dictionary mapping trigger type names to their corresponding cache file paths
        """
        gui_cache = os.path.join(PROJECT_ROOT, "lib", "gui", ".cache")
        return {"mask": os.path.join(gui_cache, ".preview_mask_toggle"),
                "refresh": os.path.join(gui_cache, ".preview_trigger")}

    @classmethod
    def _get_trainer(cls, model: ModelPlugin, batch_size: int, distributed: bool) -> TrainerPlugin:
        """ Instantiate trainer plugin (original/distributed) based on hardware configuration

        Parameters
        ----------
        model
            The loaded neural network module whose parameters will be optimized during training
        batch_size
            Number of face samples per-side processed per forward/backward pass.
        distributed
            Whether user requested distributed data parallel training

        Returns
        -------
        Instantiated trainer plugin either "original" (default) or "distributed"
        """
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
        """ Create data loader with specified augmentation options and sampling strategy

        Parameters
        ----------
        input_folders
            List of input directory paths containing extracted faces for training
        input_size
            Target image size after preprocessing - all feed images resized to this dimension
        batch_size
            Number of samples per side per training iteration
        output_shapes
            List of target shape tuples for each face encoding output (channels, height, width)
        is_rgb
            Whether input images should be loaded in RGB color order or BGR format
        augment_color
            Whether to apply random color jittering (brightness/contrast/saturation changes)
        flip
            Whether to apply random horizontal flipping during preprocessing
        warp
            Whether to apply affine transformation warping
        warp_to_landmarks : bool
            Whether to cache landmark coordinates during preprocessing for "warp-to-landmarks"
        sampler
            The sampler to use for shuffling data

        Returns
        -------
        Configured data loader instance ready to yield batches for training
        """
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
        """ Validate timelapse folder inputs exist and contain valid extracted faces

        Parameters
        ----------
        args
            Parsed command-line arguments containing timelapse_input_a, timelapse_input_b paths

        Returns
        -------
        ``True`` if validation passes (both folders exist with valid PNG images)

        Raises
        ------
        FaceswapError
            If only one timelapse folder provided instead of both required for pair-wise comparison
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

            # pylint:disable=duplicate-code
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
        """ Register optional training units based on CLI flag configuration

        Parameters
        ----------
        loop
            The training loop instance whose add_unit() method will be called for each unit
        model
            Model instance providing name and path references needed by some units
        images
            Input folder paths used by PreviewUnit to generate training previews
        args
            CLI arguments containing all flag values determining which units to register
        """
        if args.load_weights:
            if model.io.file_exists:
                logger.warning("'load_weights' selected whilst resuming an existing model. "
                               "No weights will be loaded")
            else:
                loop.add_unit(LoadWeightsUnit(args.load_weights, model))

        if args.freeze_weights:
            loop.add_unit(FreezeWeightsUnit(model))

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
        """ Build complete training loop with data loader and trainer and add all optional units

        Parameters
        ----------
        model
            Model instance providing plugin reference for trainer instantiation
        args
            CLI arguments containing trainer setup options

        Returns
        -------
        Fully configured training loop instance ready to start background thread execution
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
        """ Print startup banner with instructions for keyboard controls if running in console """
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
        """ Monitor GUI trigger files for mask toggle or refresh requests and set events

        Parameters
        ----------
        events
            Event object whose events can be set to signal to the training loop
        """
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
        """ Check for user keyboard input to control training behavior.

        Parameters
        ----------
        keypress
            Keyboard hit object providing getch() and kbhit() methods for detecting keypresses
        training_events
            Event system allowing communication between main and training threads
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
        """ Signal all background processes to terminate gracefully and reset terminal state.

        Parameters
        ----------
        keypress
            Keyboard hit object that needs to be reset back to standard terminal behavior
        """
        logger.debug("[Train] Shutting down")
        self._preview.shutdown()
        self._training_loop.events.exit.set()
        keypress.set_normal_term()

    def _monitor(self) -> None:
        """ Run interactive monitoring that coordinates preview display + user input handling """
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
                logger.info("[Train] Keyboard Interrupt Caught. Saving Weights and exiting")
                break
            except Exception as err:
                logger.error("[Train] Train Error caught: %s", str(err))
                self._shutdown(keypress)
                raise

        self._shutdown(keypress)
        logger.debug("[Train] Closed Monitor")

    def process(self) -> None:
        """ Entry point method that orchestrates complete training lifecycle from start to finish.

        Main workflow executed when training is initiated via command line or GUI. Wraps the
        TrainingLoop in a background thread, runs the interactive monitoring loop for user
        interaction and real-time feedback, then waits for completion before flushing output
        buffers and returning. Provides clean separation between setup (done in __init__),
        execution (handled here), and any cleanup that may be needed

        Execution Flow:
        ---------------
        1. Start background thread running the actual training iterations via TrainingLoop.start()
        which processes batches and updates loss metrics/plugins
        2. Enter monitoring loop for user interaction, preview display, and keyboard controls
        - Runs until exit event is set or KeyboardInterrupt received
        3. Wait for background thread to complete all queued iterations via join()
        4. Flush stdout buffer to ensure all logs are written before returning

        Notes
        -----
        The method relies on setup completed in __init__() including model loading, data loader
        configuration, trainer instantiation, and optional units registration (PreviewUnit,
        TensorBoard, Warmup, etc.). Any errors during training will propagate through exceptions
        raised by TrainingLoop or caught in monitoring loop with appropriate logging. After
        completion or early exit, all registered units have had opportunity to perform final save
        operations before thread cleanup.
        """
        logger.debug("[Train] Starting Training Process")
        self._training_loop.start()
        self._monitor()
        self._training_loop.join()
        sys.stdout.flush()
        logger.debug("[Train] Completed Training Process")


__all__ = get_module_objects(__name__)

#! /usr/env/bin/python3
"""Handles extract plugins and runners """
from __future__ import annotations

import abc
import logging
import typing as T
from queue import Queue, Empty as QueueEmpty, Full as QueueFull
from time import sleep
from uuid import uuid4

import cv2
import numpy as np
import numpy.typing as npt
from torch.cuda import OutOfMemoryError

from lib.align.constants import EXTRACT_RATIOS, LandmarkType
from lib.align.aligned_face import batch_sub_crop
from lib.multithreading import ErrorState
from lib.utils import get_module_objects, FaceswapError
from plugins.plugin_loader import PluginLoader
from plugins.extract.base import ExtractPlugin
from plugins.extract.extract_config import load_config
from .iterator import InboundIterator, InputIterator, InterimIterator, OutputIterator
from .objects import ExtractorBatch, ExtractMedia, ExtractSignal
from .runner_threads import PluginThreads


if T.TYPE_CHECKING:
    from lib.align.alignments import PNGHeaderSourceDict
    from lib.align.constants import CenteringType
    from lib.align.detected_face import DetectedFace
    from plugins.extract.base import FacePlugin

logger = logging.getLogger(__name__)
# TODO oversized batch when passthrough with lots of detections


_PLUGIN_REGISTER: dict[str, list[ExtractRunner]] = {}
"""uuid of the input runner to list of runners in the chain. Used to assert build order and when
calling the runner in passthrough mode and tracking multiple pipelines """

OOM_MESSAGE = (
    "You do not have enough GPU memory available to run detection at the selected batch size. You"
    "can try a number of things:"
    "\n1) Close any other application that is using your GPU (web browsers are particularly bad "
    "for this)."
    "\n2) Lower the batchsize (the amount of images fed into the model) by editing the plugin "
    "settings (GUI: Settings > Configure extract settings, CLI: Edit the file "
    "faceswap/config/extract.ini)."
    "\n3) Use lighter weight plugins."
    "\n4) Enable fewer plugins."
)


class ExtractRunner(abc.ABC):  # pylint:disable=too-many-instance-attributes
    """Runs an extract plugin

    Parameters
    ----------
    plugin
        The name of the plugin that this runner is to use
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    def __init__(self,
                 plugin: str,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        self._plugin_type: T.Literal["detect",
                                     "align",
                                     "mask",
                                     "identity",
                                     "dummy"] = self._get_plugin_type()
        load_config(config_file)
        self._plugin_name = plugin
        self._queues: dict[str, Queue] = {}
        self._is_first = False
        self._uuid: str | None = None
        """Unique identifier for plugin ordering and multi-plugin tracking. Populated on __call__
        to ensure a plugin is not called prior to it's input runner being called"""
        if self._plugin_type == "dummy":
            self._plugin_name = plugin
            return
        self._plugin = PluginLoader.get_extractor(self._plugin_type, plugin)
        self._plugin.compile = compile_model
        self._plugin_name = self._plugin.name
        self._dtype = self._plugin.dtype.lower()
        self._overridden = {method: self._is_overridden(method)
                            for method in ("pre_process", "process", "post_process")}
        self._threads = self._get_threads()
        self._inbound_iterator: InboundIterator | InputIterator
        self._output_iterator: OutputIterator

    @property
    def uuid(self) -> str:
        """Unique identifier for plugin ordering and multi-plugin tracking"""
        assert self._uuid is not None
        return self._uuid

    @property
    def out_queue(self) -> Queue[ExtractorBatch]:
        """The output queue from this plugin runner"""
        return self._queues["out"]

    @property
    def plugin(self) -> ExtractPlugin:
        """The plugin that this runner uses"""
        return self._plugin

    def __repr__(self) -> str:
        """Pretty print for logging"""
        return f"{self.__class__.__name__}(plugin={repr(self._plugin_name)})"

    @classmethod
    def _get_plugin_type(cls) -> T.Literal["detect", "align", "mask", "identity", "dummy"]:
        """Obtain the type of extraction plugin that this runner is responsible for

        Returns
        -------
        The type of plugin that this runner is using
        """
        plugin_type = cls.__name__.lower().replace("runner", "")
        assert plugin_type in ("detect", "align", "mask", "identity", "dummy")
        return plugin_type

    def _is_overridden(self, method_name: str) -> bool:
        """Test if a plugin method's method has been overridden

        Parameters
        ----------
        method_name
            The name of the method that is to be checked

        Returns
        -------
        ``True`` if the plugin has overridden the given method
        """
        assert self._plugin_type != "dummy"
        plugin_class = type(self._plugin)
        retval = (
            method_name in plugin_class.__dict__
            and plugin_class.__dict__[method_name] is not ExtractPlugin.__dict__.get(method_name)
            )
        logger.debug("[%s] Overridden method '%s': %s", self._plugin_name, method_name, retval)
        return retval

    def _get_threads(self) -> PluginThreads:
        """Obtain the threads required to each enabled plugin process.

        Returns
        -------
        The object that manages the threads for this plugin
        """
        assert self._plugin_type != "dummy"
        retval = PluginThreads(self._plugin_name)
        for process in self._overridden:
            logger.debug("[%s] Adding thread for '%s'", self._plugin_name, process)
            retval.register_thread(target=getattr(self, f"{process}"), name=process)
        logger.debug("[%s] Threads: %s", self._plugin_name, retval)
        return retval

    def _get_queues(self, input_runner: ExtractRunner | None) -> dict[str, Queue]:
        """Obtain the in queue to the model and the output queues from each of this plugin's
        processes

        Parameters
        ----------
        input_runner
            The input plugin or queue that feeds this plugin. ``None`` if data is to be fed
            through the runner's `put` method.

        Returns
        -------
        The plugin inbound queue and the output queue for each of this plugin's processes in
        processing order
        """
        retval: dict[str, Queue] = {}
        if self._plugin_type != "dummy":  # Just add the single queue for dummy runner
            in_queue = Queue(maxsize=1) if input_runner is None else input_runner.out_queue
            for idx, thread in enumerate(self._threads.enabled):
                queue = in_queue if idx == 0 else Queue(maxsize=1)
                logger.debug("[%s] Adding in queue for thread '%s'", self._plugin_name, thread)
                retval[thread] = queue
        logger.debug("[%s] Adding out queue", self._plugin_name)
        retval["out"] = Queue(maxsize=1)
        return retval

    def _get_inbound_iterator(self) -> InboundIterator | InputIterator:
        """Obtain the inbound iterator. If this is the first/only plugin in the pipeline, this
        will be an InputIterator that splits ExtractMedia frame objects into appropriate batches
        for the plugin.

        If this is a subsequent plugin, then an InboundIterator will be returned, which takes
        already batched data from the previous plugin and re-batches for the current plugin

        Returns
        -------
        The iterator to process inbound data for the plugin
        """
        assert self._plugin_type != "dummy"
        retval: InputIterator | InboundIterator
        if self._is_first:
            retval = InputIterator(list(self._queues.values())[0],
                                   f"{self._plugin_name}",
                                   self._plugin_type,
                                   self._plugin.batch_size)
        else:
            retval = InboundIterator(list(self._queues.values())[0],
                                     f"{self._plugin_name}",
                                     self._plugin_type,
                                     self._plugin.batch_size)
        logger.debug("[%s.in] Got inbound iterator: %s", self._plugin_name, retval)
        return retval

    def _clean_output(self,
                      batch: ExtractorBatch | ExtractSignal,
                      next_process: T.Literal["process", "post_process", "out"]) -> None:
        """Remove any images from the batch that have no detected faces and delete any internal
        plugin attributes when outputting from the plugin

        Parameters
        ----------
        batch
            The batch of data to potentially delete data from or ``None`` for EOF
        next_process
            The next process for the plugin
        """
        if next_process != "out" or isinstance(batch, ExtractSignal):
            return
        self._delete_images(batch)
        if hasattr(batch, "matrices"):
            del batch.matrices
        if hasattr(batch, "data"):
            del batch.data

    def output_info(self) -> None:
        """Called after the final item is put to the out queue. Override for plugin runner
        specific output"""
        return

    def _put_data(self, process: str, batch: ExtractorBatch | ExtractSignal) -> None:
        """Put data from a plugin's process into the next queue. If this is the first plugin in
        the pipeline and we are queueing data out from the plugin, then remove any images which
        have no detected faces.

        Parameters
        ----------
        process
            The name of the process that wishes to output data
        batch
            The batch of data to put to the next queue or an ExtractSignal after the final
            iteration
        """
        queue_names = list(self._queues)
        queue_index = queue_names.index(process) + 1
        next_process = queue_names[queue_index]
        assert next_process in ("process", "post_process", "out")
        queue = self._queues[next_process]
        self._clean_output(batch, next_process)
        logger.trace("[%s.%s] Outputting to '%s': %s",  # type:ignore[attr-defined]
                     self._plugin_name,
                     process,
                     next_process,
                     batch.name if isinstance(batch, ExtractSignal) else batch)

        while True:
            if ErrorState.has_error():
                logger.debug("[%s.%s] thread error detected. Not putting",
                             self._plugin_name, process)
                return
            try:
                logger.trace("[%s.%s] Putting to out queue: %s",  # type:ignore[attr-defined]
                             self._plugin_name,
                             process,
                             batch.name if isinstance(batch, ExtractSignal) else batch)
                queue.put(batch, timeout=1)
                break
            except QueueFull:
                logger.trace("[%s.%s] Waiting to put item",  # type:ignore[attr-defined]
                             self._plugin_name, process)
                continue

        if next_process == "out" and isinstance(batch, ExtractSignal):
            sleep(1)  # Wait for downstream plugins to flush
            self.output_info()

    def _handle_zero_detections(self, process, batch: ExtractorBatch) -> bool:
        """Check if the given batch has detected faces. If not, pass it straight through to the
        next queue

        Parameters
        ----------
        process
            The name of the process that is checking for zero detections
        batch
            The batch of data to check for zero detections

        Returns
        -------
        ``True`` if the batch has no face detections and has been passed on. ``False`` if the batch
        contains data to be processed
        """
        if self._plugin_type == "detect" or batch.frame_ids.size:
            return False
        logger.trace(  # type:ignore[attr-defined]
            "[%s.%s] Passing through batch with no detections",  self._plugin_name, process)
        self._put_data(process, batch)
        return True

    def _get_data(self, process: str) -> T.Generator[ExtractorBatch, None, None]:
        """Get the next batch of data for the thread's process."""
        assert self._plugin_type != "dummy"
        queue = self._queues[process]
        name = f"{self._plugin_name}_{process}"
        if list(self._queues).index(process) == 0:
            iterator: InboundIterator | InputIterator | InterimIterator = self._inbound_iterator
        else:
            iterator = InterimIterator(queue, name, self._plugin_type, self._plugin.batch_size)
        for batch in iterator:
            if batch == ExtractSignal.FLUSH:  # pass flush downstream
                self._put_data(process, batch)
                continue
            assert isinstance(batch, ExtractorBatch)
            if self._handle_zero_detections(process, batch):
                continue
            yield batch

    def _delete_images(self, batch: ExtractorBatch) -> None:
        """Delete any images from the batch where there are no faces

        Parameters
        ----------
        batch
            The batch of data to delete images without faces from
        """
        no_boxes = [i for i in range(len(batch.images)) if i not in batch.frame_ids]
        if not no_boxes:
            return
        logger.trace(  # type:ignore[attr-defined]
            "[%s.out] Deleting %s of %s images with no bounding boxes",
            self._plugin_name, len(no_boxes), len(batch.images))
        for idx in no_boxes:
            batch.images[idx] = np.empty(shape=(0, 0, 3), dtype="uint8")

    def _format_images(self, images: npt.NDArray[np.uint8]) -> np.ndarray:
        """Format the incoming UINT8 0-255 images to the format specified by the plugin

        Parameters
        ----------
        images
            The batch of UINT8 images to format

        Returns
        -------
        The batch of images formatted and scaled for the plugin
        """
        retval = images if self._dtype == "uint8" else images.astype(self._dtype)
        if self._plugin.scale == (0, 255):
            return retval
        low, high = self._plugin.scale
        im_range = high - low
        retval /= (255. / im_range)
        retval += low
        return retval

    @abc.abstractmethod
    def pre_process(self) -> None:
        """Override for plugin type runner specific behavior"""

    def _predict(self, feed: np.ndarray) -> np.ndarray:
        """Obtain a prediction from the plugin

        Parameters
        ----------
        feed
            The batch to feed the model

        Returns
        -------
        The prediction from the model

        Raises
        ------
        FaceswapError
            If an OOM occurs
        """
        feed_size = feed.shape[0]
        is_padded = self._plugin.compile and feed_size < self._plugin.batch_size
        batch_feed = feed
        if is_padded:  # Prevent model re-compile on undersized batch
            batch_feed = np.empty((self._plugin.batch_size, *feed.shape[1:]), dtype=feed.dtype)
            logger.debug("[%s.process] Padding undersized batch of shape %s to %s",
                         self._plugin.name, feed.shape, batch_feed.shape)
            batch_feed[:feed_size] = feed
        try:
            retval = self._plugin.process(batch_feed)
        except OutOfMemoryError as err:
            raise FaceswapError(OOM_MESSAGE) from err
        if is_padded and retval.dtype == "object":
            out = np.empty(retval.shape, dtype="object")
            out[:] = [x[:feed_size] for x in retval]
            retval = out
        elif is_padded:
            retval = retval[:feed_size]
        return retval

    def process(self) -> None:
        """Perform inference to get results from the plugin and pass the batch to the next process'
        queue. Override for runner specific processing"""
        process = "process"
        logger.debug("[%s.%s] Loading model", self._plugin_name, process)
        self._plugin.model = self._plugin.load_model()  # Load here to keep Cuda in same thread
        logger.debug("[%s.%s] Starting process", self._plugin_name, process)
        for batch in self._get_data(process):
            batch.data = self._predict(batch.data)
            self._put_data(process, batch)
        logger.debug("[%s.%s] Finished process", self._plugin_name, process)
        self._put_data(process, ExtractSignal.SHUTDOWN)

    @abc.abstractmethod
    def post_process(self) -> None:
        """Override for plugin type runner specific behavior"""

    def _put_to_input(self, data: ExtractMedia | ExtractorBatch | ExtractSignal) -> None:
        """Put data to the runner's input queue, monitoring for errors

        Parameters
        ----------
        data
            The object to put into the runner's in queue
        """
        while True:
            if ErrorState.has_error():
                raise RuntimeError("Error in thread")
            try:
                self._queues[list(self._queues)[0]].put(data, timeout=1)
                break
            except QueueFull:
                logger.debug("[%s] Waiting on queue", self._plugin_name)
                continue

    @T.overload
    def put(self,
            filename: str,
            image: npt.NDArray[np.uint8],
            detected_faces: list[DetectedFace] | None = None,
            source: str | None = None,
            is_aligned: bool = False,
            frame_metadata: PNGHeaderSourceDict | None = None,
            passthrough: T.Literal[False] = False) -> None: ...

    @T.overload
    def put(self,
            filename: str,
            image: npt.NDArray[np.uint8],
            detected_faces: list[DetectedFace] | None = None,
            source: str | None = None,
            is_aligned: bool = False,
            frame_metadata: PNGHeaderSourceDict | None = None,
            *,
            passthrough: T.Literal[True]) -> ExtractMedia: ...

    def put(self,
            filename: str,
            image: npt.NDArray[np.uint8],
            detected_faces: list[DetectedFace] | None = None,
            source: str | None = None,
            is_aligned: bool = False,
            frame_metadata: PNGHeaderSourceDict | None = None,
            passthrough: bool = False) -> None | ExtractMedia:
        """Put a frame into the pipeline.

        Note
        ----
        When a pipeline is built using the __call__ method, this method will always put items into
        the first plugin in the pipeline

        Parameters
        ----------
        filename
            The filename of the frame
        image
            The loaded frame as UINT8 BGR array
        detected_faces
            The detected face objects for the frame. ``None`` if not any. Default: ``None``
        source
            The full path to the source folder or video file. Default: ``None`` (Not provided)
        is_aligned
            ``True`` if the image being passed into the pipeline is an aligned faceswap face.
            Default: ``False``
        frame_metadata
            If the image is aligned then the original frame metadata can be added here. Some
            plugins (eg: mask) require this to be populated for aligned inputs. Default: ``None``
        passthrough
            ``True`` if this item is meant to be passed straight through the extraction pipeline
            with no caching, for immediate return. Default: ``False``

        Returns
        -------
        If passthrough is ``True`` returns the output ExtractMedia object, otherwise ``None``
        """
        item = ExtractMedia(filename=filename,
                            image=image,
                            detected_faces=detected_faces,
                            source=source,
                            is_aligned=is_aligned,
                            passthrough=passthrough)
        if frame_metadata is not None:
            item.add_frame_metadata(frame_metadata)
        self._put_to_input(item)
        if passthrough:
            return next(_PLUGIN_REGISTER[self.uuid][-1])
        return None

    def put_media(self, media: ExtractMedia) -> None | ExtractMedia:
        """Put a frame into the pipeline that is within an ExtractMedia object.

        Note
        ----
        When a pipeline is built using the __call__ method, this method will always put items into
        the first plugin in the pipeline

        Parameters
        ----------
        media
            The ExtractMedia object to put into the pipeline

        Returns
        -------
        If the ExtractMedia's passthrough is ``True`` returns the output ExtractMedia object,
        otherwise ``None``
        """
        self._put_to_input(media)
        if media.passthrough:
            return next(_PLUGIN_REGISTER[self.uuid][-1])
        return None

    def put_direct(self,  # noqa[C901]
                   filename: str,
                   image: npt.NDArray[np.uint8],
                   detected_faces: list[DetectedFace],
                   is_aligned: bool = False,
                   frame_size: tuple[int, int] | None = None) -> ExtractorBatch:
        """Put an item directly into this runner's plugin and return the result

        Parameters
        ----------
        filename
            The filename of the frame
        image
            The loaded frame as UINT8 BGR array
        detected_faces
            The detected face objects for the frame
        is_aligned
            ``True`` if the image being passed into the pipeline is an aligned faceswap face.
            Default: ``False``
        frame_size
            The (height, width) size of the original frame if passing in an aligned image

        Raises
        ------
        ValueError
            If attempting to put an ExtractorBatch object to the first runner in the pipeline or if
            providing an aligned image with insufficient data

        Returns
        -------
        ExtractorBatch
            The output from this plugin for the given input
        """
        if isinstance(self._inbound_iterator, InputIterator):
            raise ValueError("'put_direct' should not be used on the first runner in a "
                             "pipeline. Use the runner's `put` method")
        if self._plugin_type not in ("detect", "align") and not is_aligned:
            raise ValueError(f"'{self._plugin_type}' requires aligned input")
        if self._plugin_type in ("detect", "align") and is_aligned:
            raise ValueError(f"'{self._plugin_type}' requires non-aligned input")
        if is_aligned and not frame_size:
            raise ValueError("Aligned input must provide the original frame_size")
        batch = ExtractorBatch(filenames=[filename],
                               images=[image],
                               is_aligned=is_aligned)
        batch.bboxes = np.array([[f.left, f.top, f.right, f.bottom]
                                 for f in detected_faces], dtype="int32")
        batch.frame_ids = np.zeros((batch.bboxes.shape[0], ), dtype="int32")
        batch.frame_sizes = [frame_size] if frame_size else None
        if self._plugin_type not in ("detect", "align"):
            landmarks = np.array([f.landmarks_xy for f in detected_faces], dtype="float32")
            batch.landmarks = landmarks
            batch.landmark_type = LandmarkType.from_shape(T.cast(tuple[int, int],
                                                                 landmarks.shape[1:]))
        original_out = self._queues["out"]  # Unhook queue from next runner
        self._queues["out"] = Queue(maxsize=1)
        self._put_to_input(batch)
        self._put_to_input(ExtractSignal.FLUSH)

        result: list[ExtractorBatch] = []
        while True:
            if ErrorState.has_error():
                ErrorState.check_and_raise()
            try:
                out = self._queues["out"].get(timeout=0.25)
            except QueueEmpty:
                continue
            if out == ExtractSignal.FLUSH:
                break
            result.append(out)

        self._queues["out"] = original_out  # Re-attach queue to next runner

        retval = result[0]
        if len(result) > 1:
            for remain in result[1:]:
                retval.append(remain)
        return retval

    def stop(self) -> None:
        """Indicate to the runner that there is no more data to be ingested"""
        logger.debug("[%s] Putting EOF to runner", self._plugin_name)
        self._put_to_input(ExtractSignal.SHUTDOWN)
        logger.debug("[%s] Removing pipeline '%s'", self._plugin.name, self.uuid)
        del _PLUGIN_REGISTER[self.uuid]

    def flush(self) -> None:
        """Flush all data currently within the pipeline"""
        logger.debug("[%s] Putting FLUSH to runner", self._plugin_name)
        self._put_to_input(ExtractSignal.FLUSH)

    def __iter__(self) -> T.Self:
        """This is an iterator"""
        return self

    def __next__(self) -> ExtractMedia:
        """Obtain the next item from the plugin's output

        Returns
        -------
        The media object with populated detected faces for a frame
        """
        retval = next((self._output_iterator), None)
        ErrorState.check_and_raise()
        if retval is None:
            raise StopIteration
        return retval

    def _cascade_interfaces(self, input_runner: ExtractRunner | None) -> None:
        """On this runner's call method, cascade the public interfaces to be the input runner's
        public interfaces, such that calling them from the final plugin in the pipeline actually
        interacts with the first plugin in the pipeline

        Parameters
        ----------
        input_runner
            The input runner to this runner or ``None`` if this is the first runner in the pipeline
        """
        if input_runner is None:
            return
        setattr(self, "put", input_runner.put)
        setattr(self, "put_media", input_runner.put_media)
        setattr(self, "stop", input_runner.stop)
        setattr(self, "flush", input_runner.flush)

        logger.debug("[%s] Set pipeline interfaces to %s",
                     self.__class__.__name__,
                     [f"{f.__self__.__class__.__name__}.{f.__func__.__name__}"
                      for f in (self.put, self.put_media, self.stop, self.flush)])

    def _register_plugin(self, input_runner: ExtractRunner | None = None) -> None:
        """Register the plugin into the plugin tracker

        Parameters
        ----------
        input_runner
            The input plugin that feeds this plugin or ``None`` if data is to be fed through the
            runner's `put` method. Default: ``None``
        """
        name = f"{self.__class__.__name__}.{self._plugin_name}"
        if input_runner is None:
            logger.debug("[%s] Registering new pipeline: '%s'", name, self.uuid)
            _PLUGIN_REGISTER[self.uuid] = [self]
            return
        uid, chain = next((k, v) for k, v in _PLUGIN_REGISTER.items() if input_runner in v)
        logger.debug("[%s] Adding to existing pipeline: '%s'", name, uid)
        chain.insert(chain.index(input_runner) + 1, self)

    def __call__(self, input_runner: ExtractRunner | None = None) -> T.Self:
        """Build and start the plugin runner

        Parameters
        ----------
        input_runner
            The input plugin that feeds this plugin or ``None`` if data is to be fed through the
            runner's `put` method. Default: ``None``

        Returns
        -------
        The extract plugin runner that has been called

        Raises
        ------
        ValueError
            If the input runner has not been called and assigned a UUID or if this runner has
            already been called
        """
        if input_runner is not None and input_runner._uuid is None:
            raise ValueError(f"Input runner '{input_runner.__class__.__name__}' must be called "
                             f"prior to adding to '{self.__class__.__name__}'")
        if self._uuid is not None:
            raise ValueError(f"Runner '{self.__class__.__name__}' has already been called")
        self._uuid = uuid4().hex

        self._is_first = input_runner is None
        self._queues = self._get_queues(input_runner)

        if self._plugin_type == "dummy":
            if input_runner is not None:
                raise ValueError("Dummy plugin cannot have an input runner")
            logger.debug("[%s] Returning early for dummy plugin call", self.__class__.__name__)
            return self
        self._inbound_iterator = self._get_inbound_iterator()
        self._output_iterator = OutputIterator(self._queues["out"],
                                               f"{self._plugin_name}_out",
                                               self._plugin_type,
                                               self._plugin.batch_size)
        self._cascade_interfaces(input_runner)
        self._register_plugin(input_runner)
        self._threads.start()
        return self


class ExtractRunnerFace(ExtractRunner, abc.ABC):
    """Runs an extract plugin. Extended with methods common to plugins that use aligned face
    images as input

    Parameters
    ----------
    plugin
        The name of the plugin that this runner is to use
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    _logged_warning: dict[str, bool] = {"mask": False, "identity": False}
    """Stores whether a warning has been issued for non-68 point landmarks for this plugin type"""

    def __init__(self,
                 plugin: str,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        super().__init__(plugin, compile_model=compile_model, config_file=config_file)
        self._plugin: FacePlugin

        self._input_size = self._plugin.input_size
        self._centering: CenteringType = self._plugin.centering
        self.storage_name = self._plugin.storage_name
        """The name that the object will be stored with in the alignments file"""

        self._padding = round((self._input_size * EXTRACT_RATIOS[self._centering]) / 2)
        self._aligned_mat_name = ("matrices" if self._centering == "legacy"
                                  else f"matrices_{self._centering}")

        # Aligned handling
        self._head_to_base_ratio = (1 - EXTRACT_RATIOS["head"]) / 2
        self._head_to_centering_ratio = ((1 - EXTRACT_RATIOS["head"]) /
                                         (1 - EXTRACT_RATIOS[self._centering]) / 2)
        self._aligned_offsets_name = f"offsets_{self._centering}"

    @property
    def plugin(self) -> FacePlugin:
        """The plugin that this runner uses"""
        return self._plugin

    def _maybe_log_warning(self, landmark_type: LandmarkType | None) -> None:
        """Log a warning the first time if/when non-68 point landmarks are seen

        Parameters
        ----------
        landmark_type
            The type of landmarks within the batch
        """
        assert landmark_type is not None
        if self._logged_warning[self._plugin_type] or landmark_type == LandmarkType.LM_2D_68:
            return
        ptype = "Masks" if self._plugin_type == "mask" else "Identities"
        logger.warning("Faces do not contain landmark data. %s are likely to be sub-standard",
                       ptype)
        self._logged_warning[self._plugin_type] = True

    # Pre-processing
    def _get_matrices(self, matrices: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Obtain the (N, 2, 3) matrices for the face plugin's centering type

        Parameters
        ----------
        matrices
            The normalized alignment matrices for aligning faces from the image

        Returns
        -------
        The adjustment matrices for taking the image patch from the image for plugin input
        """
        mats = matrices[:, :2] * (self._input_size - 2 * self._padding)
        mats[:, :, 2] += self._padding
        return mats

    def _get_faces(self,  # pylint:disable=too-many-locals
                   images: list[npt.NDArray[np.uint8]],
                   image_ids: npt.NDArray[np.int32],
                   matrices: npt.NDArray[np.float32],
                   with_alpha: bool = False) -> npt.NDArray[np.uint8]:
        """Obtain the cropped and aligned faces from the batch of images

        Parameters
        ----------
        images
            The full size frames for the batch
        image_ids
            The image ids for each detected face
        matrices
            The adjustment matrices for taking the image patch from the frame for plugin input
        with_alpha
            ``True`` to add a filled alpha channel to the batch of images prior to warping to
            faces. Default: ``False``

        Returns
        -------
        Batch of 3 or 4 channel face patches for feeding the model. If `with_alpha` is selected
        then the final channel is an ROI mask indicating areas that go out of bounds
        """
        scales = np.hypot(matrices[..., 0, 0], matrices[..., 1, 0])  # Always same x/y scaling
        interpolations = np.where(scales > 1.0, cv2.INTER_LINEAR, cv2.INTER_AREA)
        size = (self._input_size, self._input_size)
        channels = 4 if with_alpha else 3
        retval = np.zeros((len(image_ids), *size, channels), dtype=images[image_ids[0]].dtype)

        for idx, (image_id, mat, interpolation) in enumerate(zip(image_ids,
                                                                 matrices,
                                                                 interpolations)):
            img: np.ndarray = images[image_id]
            if with_alpha:
                alpha = np.ones((*img.shape[:2], 1), dtype=img.dtype) * 255
                img = np.concatenate([img, alpha], axis=-1)
            cv2.warpAffine(img, mat, size, dst=retval[idx], flags=interpolation)
        return retval

    # Aligned faces as input methods
    def _batch_resize(self,
                      images: npt.NDArray[np.uint8],
                      destination: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Resize a batch of images of the same dimensions into the given destination array

        Parameters
        ----------
        images
            The batch of images to be resized
        destination
            The destination array to place the resized images into
        interpolation
            The interpolation to use for the resize operation

        Returns
        -------
        The resized images
        """
        in_size = images.shape[1]
        if in_size == self._input_size:
            return images

        interpolation = cv2.INTER_AREA if self._input_size < in_size else cv2.INTER_LINEAR
        out_size = (self._input_size, self._input_size)
        for idx, img in enumerate(images):
            cv2.resize(img, out_size, dst=destination[idx], interpolation=interpolation)
        return destination

    def _get_faces_aligned(self,
                           images: list[npt.NDArray[np.uint8]],
                           image_ids: npt.NDArray[np.int32],
                           source_padding: npt.NDArray[np.float32],
                           dest_padding: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """Obtain the batch of faces when input images are a batch of extracted faceswap faces

        Parameters
        ----------
        images
            The batch of faceswap extracted faces to obtain the model input images from
        image_ids
            The image ids for each detected face
        source_padding
            The normalized (N, x, y) padding used for the aligned image's centering
        dest_padding
            The normalized (N, x, y) padding used for the plugin's centering

        Returns
        -------
        The sub-crop from the aligned faces for feeding the model
        """
        imgs = np.array([images[idx] for idx in image_ids] if len(images) != len(image_ids)
                        else images)
        assert imgs.dtype != object, "Aligned images must all be the same size"
        dst = np.empty((imgs.shape[0], self._input_size, self._input_size, 3), dtype=imgs.dtype)
        if self._centering == "head":
            return self._batch_resize(imgs, dst)

        src_size = imgs.shape[1]
        out_size = 2 * int(np.rint(src_size * self._head_to_centering_ratio))
        base_size = 2 * int(np.rint(src_size * self._head_to_base_ratio))
        padding_diff = (src_size - out_size) // 2
        delta = dest_padding - source_padding
        offsets = np.rint(delta * base_size + padding_diff).astype("int32")
        imgs = batch_sub_crop(imgs, offsets, out_size)
        return self._batch_resize(imgs, dst)


__all__ = get_module_objects(__name__)

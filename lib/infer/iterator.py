#! /usr/env/bin/python3
""" Iterators for ingesting into and passing data through extract plugin runners """

from __future__ import annotations

import abc
import logging
import typing as T

from queue import Queue, Empty as QueueEmpty

import numpy as np

from lib.align.aligned_mask import Mask
from lib.align.detected_face import DetectedFace
from lib.infer.objects import ExtractMedia, ExtractSignal
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from lib.multithreading import ErrorState

from .objects import ExtractorBatch

logger = logging.getLogger(__name__)
QueueItemInT = T.TypeVar("QueueItemInT")
QueueItemOutT = T.TypeVar("QueueItemOutT")


class ExtractIterator(T.Generic[QueueItemInT, QueueItemOutT], abc.ABC):
    """Base class for iterators within Faceswap's extract pipeline

    Type Parameters
    ---------------
    QueueItemInT
        Type of item received from the input queue.

    QueueItemOutT
        Type yielded by the iterator.

    Parameters
    ----------
    queue
        The inbound queue to the plugin
    name
        The plugin name and process calling this iterator
    plugin_type
        The type of extractor plugin that this iterator is serving
    batch_size
        The batch size that data should be returned from the iterator
    """
    def __init__(self,
                 queue: Queue[QueueItemInT | ExtractSignal],
                 name: str,
                 plugin_type: T.Literal["detect", "align", "mask", "identity"],
                 batch_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._queue = queue
        self._name = f"{name}.{self.__class__.__name__.replace('Iterator', '').lower()}"
        self._plugin_type = plugin_type
        self._fifo: list[QueueItemOutT] = []
        self._batch_size = batch_size
        self._zero_detect_threshold = batch_size * 2
        self._flush = False
        self._shutdown = False

    def __iter__(self) -> T.Self:
        """ This is an iterator """
        return self

    def __repr__(self) -> str:
        """ Pretty print for logging """
        params = {k[1:]: repr(v)
                  for k, v in self.__dict__.items()
                  if k in ("_queue",
                           "_batch_size",
                           "_name",
                           "_plugin_type")}
        sparams = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({sparams})"

    def _from_queue(self) -> QueueItemInT | ExtractSignal | None:
        """ Get the next item from the queue on a 1 second timeout.

        Returns
        -------
        ExtractorBatch or ExtractMedia or ExtractSignal or None
            The next item from the queue or ``None`` if no item is available
        """
        try:
            retval = self._queue.get(timeout=1)
        except QueueEmpty:
            logger.trace("[%s] No item available", self._name)  # type:ignore[attr-defined]
            return None
        logger.trace("[%s] From queue: %s",  # type:ignore[attr-defined]
                     self._name, retval.name if isinstance(retval, ExtractSignal) else retval)
        return retval

    def _has_zero_detections(self) -> bool:
        """If we are an input or inbound iterator and we have an item in FIFO that contains a lot
        of zero detections (plugin batchsize * 3) then release the item to prevent stacking frames
        into RAM and long phases when nothing is being output from the pipeline.

        Returns
        -------
        ``True`` if there are a lot of frames with no detections in the FIFO
        """
        if len(self._fifo) != 1:  # Any more than 1 and batch will be output anyway
            return False
        if self._plugin_type == "detect":
            return False  # Will always be 0 detections for detect and will never hit threshold
        if self.__class__.__name__ not in ("InputIterator", "InboundIterator"):
            return False  # We only care about inputs to runners
        item = T.cast(ExtractorBatch, self._fifo[0])
        zero_detects = len(item.filenames) - len(set(item.frame_ids))
        return zero_detects >= self._zero_detect_threshold

    def _from_fifo(self) -> QueueItemOutT | None:
        """ Pop the next item available in the fifo list. 1 item always remains in the list for
        appending to and should be flushed at the last iteration

        Returns
        -------
        ExtractorBatch or ExtractMedia or ExtractSignal or None
            The next available item or ``None`` if no items are available
        """
        if not self._fifo:
            logger.trace("[%s.fifo] FIFO empty",  self._name)  # type:ignore[attr-defined]
            return None
        if self._has_zero_detections():
            retval = self._fifo.pop(0)
            logger.debug("[%s.fifo] Popping from FIFO due to accumulated zero detections "
                         "(frames: %s, faces: %s)", self._name,
                         len(T.cast(ExtractorBatch, retval).filenames),
                         len(T.cast(ExtractorBatch, retval).frame_ids))
            return retval
        if len(self._fifo) <= 1:
            logger.trace("[%s.fifo] No items available. batches: %s",  # type:ignore[attr-defined]
                         self._name, len(self._fifo))
            return None
        retval = self._fifo.pop(0)
        logger.trace("[%s.fifo] Popping: %s",  # type:ignore[attr-defined]
                     self._name, retval)
        return retval

    def _handle_signals(self) -> ExtractSignal | None:
        """ Check if :attr:`_flush` or :attr:`_eof` have been set. If so, log and reset them. If
        flush has been set return the FLUSH enum

        Returns
        -------
        :class:`lib.extract.objects.ExtractSignal` | None
            The flush enum, if the iterator has received a flush signal or ``None`` if it has not

        Raises
        ------
        StopIteration
            If EOF has been seen
        """
        if self._shutdown:
            self._shutdown = False
            logger.debug("[%s] EOF Executed", self._name)
            raise StopIteration

        if not self._flush:
            return None

        self._flush = False
        logger.debug("[%s] sending FLUSH downstream", self._name)
        return ExtractSignal.FLUSH

    def _handle_inbound_signal(self, inbound: ExtractSignal) -> QueueItemOutT | ExtractSignal:
        """ Handle any received signals from the queue

        Parameters
        ----------
        inbound
            An inbound item from an iterator's in queue

        Returns
        -------
        ExtractorBatch or ExtractMedia or ExtractSignal or None
            The inbound item from the iterator's in queue if it is not a signal or if there are
            items queued for output

        Raises
        ------
        StopIteration
            If a shutdown signal has been received and there are no items queued for output
        """
        signal = inbound.name
        logger.debug("[%s] %s received. FIFO size: %s", self._name, signal, len(self._fifo))

        if self._fifo:
            setattr(self, f"_{signal.lower()}", True)
            assert len(self._fifo) == 1  # Final batch should remain
            retval = self._fifo.pop(0)
            logger.debug("[%s] Returning final queued output item: %s", self._name, retval)
            return retval

        if inbound == ExtractSignal.SHUTDOWN:
            logger.debug("[%s] SHUTDOWN Executed", self._name)
            raise StopIteration

        return inbound

    def _check_error(self) -> None:
        """ Check whether there has been a thread error and stop iteration if so

        Raises
        ------
        StopIteration
            If a thread error has been detected
        """
        if ErrorState.has_error():
            logger.debug("[%s] Thread error received", self._name)
            raise StopIteration

    @abc.abstractmethod
    def __next__(self) -> QueueItemOutT | ExtractSignal:
        """ Override to return the next batch item from the iterator

        Returns
        -------
        ExtractorBatch or ExtractMedia or ExtractSignal
            Batch object for pipeline processing, or a final media object
            when exiting the pipeline.
        """


class InputIterator(ExtractIterator[ExtractMedia, ExtractorBatch]):
    """ An iterator that processes ExtractMedia data that is input to a plugin pipeline to create
    ExtractorBatch objects at the correct batch size for processing through the pipeline's first
    plugin

    Parameters
    ----------
    queue
        The inbound queue to the plugin pipeline
    name
        The plugin name and process calling this iterator
    plugin_type
        The type of extractor plugin that this iterator is serving
    batch_size
        The batch size that data should be returned from the iterator
    """
    def _append_to_fifo(self, batch: ExtractorBatch) -> None:
        """ Append batch items to :attr:`_fifo` when it is either empty, or the last item in the
        FIFO is the correct batch size

        Adds the batch object to FIFO splitting to the plugin's batch size if required

        Parameters
        ----------
        batch
            The data from the inbound ExtractMedia object placed into an ExtractorBatch object
        """
        num_boxes = batch.bboxes.shape[0]
        if num_boxes <= self._batch_size:
            # If this is a detection plugin then boxes will always be 0, but there will only ever
            # be a single frame, so this test is fine for both detection + face plugins
            self._fifo.append(batch)
            logger.trace("[%s] Added to FIFO: %s", self._name, batch)  # type:ignore[attr-defined]
            return

        i = 0
        while i < num_boxes:
            end = i + self._batch_size
            self._fifo.append(batch[i:end])
            i += self._fifo[-1].bboxes.shape[0]
        logger.trace(  # type:ignore[attr-defined]
            "[%s] Split batch with %s boxes to FIFO boxes of size: %s",
            self._name, num_boxes, [b.bboxes.shape[0] for b in self._fifo])

    def _media_to_extractor_batch(self, media: ExtractMedia) -> ExtractorBatch:
        """Convert an incoming ExtractMedia object to an ExtractorBatch object

        Parameters
        ----------
        media
            The inbound ExtractMedia object

        Returns
        -------
        The converted ExtractorBatch object

        Raises
        ------
        RuntimeError
            When a passthrough media object is passed in, but there are items queued
        """
        if media.passthrough and self._fifo:
            raise RuntimeError("Pipeline must be empty when adding a passthrough object")
        retval = ExtractorBatch(
            [media.filename],
            [media.image],
            sources=[media.source],
            is_aligned=media.is_aligned,
            frame_sizes=[media.image_size] if media.is_aligned else None,
            frame_metadata=[media.frame_metadata] if media.frame_metadata else None,
            passthrough=media.passthrough
        )
        if media.detected_faces:
            retval.from_detected_faces(media.detected_faces)
        return retval

    def _add_data_to_batch(self, media: ExtractMedia) -> None:
        """ Add the incoming ExtractMedia data to the either the last existing extractor batch
        object or a new one.

        Parameters
        ----------
        media
            The incoming frame data

        Raises
        ------
        ValueError
            If aligned and non-aligned images are added to the same extractor batch
        """
        in_batch = self._media_to_extractor_batch(media)
        if not self._fifo:  # Add straight in to a fresh FIFO
            self._append_to_fifo(in_batch)
            return

        last_fifo = self._fifo[-1]
        exist_size = (len(last_fifo.filenames)
                      if self._plugin_type == "detect"
                      else len(last_fifo.bboxes))

        if exist_size == self._batch_size:  # Append straight onto the end of FIFO
            self._append_to_fifo(in_batch)
            return

        capacity = self._batch_size - exist_size
        num_boxes = in_batch.bboxes.shape[0]
        to_add = len(in_batch.filenames) if self._plugin_type == "detect" else num_boxes

        if media.is_aligned != last_fifo.is_aligned:
            raise ValueError("Mixing aligned and non-aligned images is not supported")

        if to_add <= capacity:  # Append to the last item in the FIFO
            last_fifo.append(in_batch)
            logger.trace(  # type:ignore[attr-defined]
                "[%s] Added batch with %s items to existing batch of %s items",
                self._name, to_add, exist_size)
            return

        # Only ExtractMedia containing detected faces that need to be added to the last item in the
        # fifo and then subsequently splitting will exist here
        split_batch = in_batch[0:capacity]
        last_fifo.append(split_batch)
        logger.trace(  # type:ignore[attr-defined]
            "[%s] Added batch with %s items to existing batch of %s items",
            self._name, capacity, exist_size)
        self._append_to_fifo(in_batch[capacity:capacity + (num_boxes - capacity)])

    def __next__(self) -> ExtractorBatch | ExtractSignal:
        """ Get the next batch of data from the iterator. Depending on the plugin type calling this
        iterator, a batch object will be returned for the given batch size of frames (for detect
        plugins) or faces (for all other plugins)

        Returns
        -------
        ExtractorBatch or ExtractSignal
            A new Batch object containing the batch to process through the plugin or a signal

        Raises
        ------
        StopIteration
            When the input is exhausted
        """
        flush = self._handle_signals()
        if flush:
            return flush

        while True:
            self._check_error()
            retval = self._from_fifo()
            if retval:
                return retval

            media = self._from_queue()
            if media is None:
                continue

            if isinstance(media, ExtractSignal):
                return self._handle_inbound_signal(media)

            if media.passthrough:
                return self._media_to_extractor_batch(media)

            self._add_data_to_batch(media)


class InboundIterator(ExtractIterator[ExtractorBatch, ExtractorBatch]):
    """ An iterator that processes ExtractorBatch data from a previous plugin and configures it as
    an input for the current plugin.

    An Inbound iterator assumes that the plugin's batch size are the number of faces (not frames)
    that it can process at one time. Detect plugins are the only plugins that work with frames
    rather than faces, but these will always be the input to the pipeline, so will use an
    InputIterator not an InboundIterator

    Parameters
    ----------
    queue
        The outbound queue from the previous plugin
    name
        The plugin name and process calling this iterator
    plugin_type
        The type of extractor plugin that this iterator is serving
    batch_size
        The batch size that data should be returned from the iterator
    """
    def _batch_to_fifo(self, in_batch: ExtractorBatch) -> None:
        """ Batch the incoming data into an object batched for the current plugin's batch size and
        add to :attr:`_fifo`

        Parameters
        ----------
        in_batch
            The inbound batch to be re-batched for output
        """
        if self._fifo and (self._fifo[-1].bboxes.shape[0] != self._batch_size):
            # Partially filled batch is queued or we are appending frames with no detections
            batch = self._fifo[-1]
            logger.trace(  # type:ignore[attr-defined]
                "[%s] Adding %s face(s) from %s image(s) to partial batch with %s face(s)",
                self._name, len(in_batch.bboxes), len(in_batch.images), len(batch.bboxes))
            batch.append(in_batch)
            return

        logger.trace("[%s] Adding new batch for %s face(s)",  # type:ignore[attr-defined]
                     self._name, len(in_batch.bboxes))
        self._fifo.append(in_batch)

    def _handle_non_split_batch(self, batch: ExtractorBatch) -> tuple[int, int]:
        """Pass inbound batches with either no boxes or the exact number of boxes required to fill
        the next batch straight through

        Parameters
        ----------
        batch
            The inbound batch to check and potentially pass straight through

        Returns
        -------
        num_boxes
            The number of boxes that exist within the inbound batch
        capacity
            The number of free slots in the next outbound batch
        """
        partial = self._fifo and self._fifo[-1].bboxes.shape[0] != self._batch_size
        num_boxes = batch.bboxes.shape[0]
        capacity = (self._batch_size - self._fifo[-1].bboxes.shape[0]
                    if partial else self._batch_size)
        if num_boxes not in (0, capacity):  # Batch needs splitting
            return num_boxes, capacity

        self._batch_to_fifo(batch)
        logger.trace(  # type:ignore[attr-defined]
            "[%s] Passed non-split batch straight through %s(frames=%s, faces=%s) to: %s"
            "(frames=%s, faces=%s)",
            self._name,
            batch.__class__.__name__,
            len(batch.filenames),
            num_boxes,
            self._fifo[-1].__class__.__name__,
            len(self._fifo[-1].filenames),
            self._fifo[-1].bboxes.shape[0])
        return 0, 0

    def _append_no_boxes(self, batch: ExtractorBatch) -> None:
        """ Incoming batches will only be processed until the last frame containing a face. Append
        any frames at the end of the incoming batch, that do not contain any faces, to the last
        queued batch

        Parameters
        ----------
        batch
            The inbound batch to append frames without boxes
        """
        start = batch.frame_ids[-1] + 1
        if start >= len(batch.filenames):
            return
        logger.trace(  # type:ignore[attr-defined]
            "[%s] Appending %s frames without faces to last batch",
            self._name, len(batch.filenames[start:]))
        self._batch_to_fifo(ExtractorBatch(batch.filenames[start:],
                                           batch.images[start:],
                                           batch.sources[start:]))

    def _rebatch_data(self, batch: ExtractorBatch) -> None:  # pylint:disable=too-many-locals
        """ Process the incoming batch data and re-batch it for the requested plugin batch size
        into the correct object and store in :attr:`_fifo`

        Parameters
        ----------
        batch
            The incoming batch of data to this plugin at the batch size of the previous plugin
        """
        num_boxes, capacity = self._handle_non_split_batch(batch)
        if num_boxes == 0:
            return

        i = count = 0
        while i < num_boxes:
            end = i + capacity
            in_batch = batch[i:end]
            self._batch_to_fifo(in_batch)
            i += len(in_batch.bboxes)
            capacity = self._batch_size  # New full batch object
            count += 1

        self._append_no_boxes(batch)
        logger.trace(  # type:ignore[attr-defined]
            "[%s] Rebatched %s, %s(frames=%s, faces=%s) to: %s",
            self._name,
            batch.filenames,
            batch.__class__.__name__,
            len(batch.filenames),
            batch.bboxes.shape[0],
            ", ".join(f"{b.__class__.__name__}(frames={len(b.filenames)}, "
                      f"faces={b.bboxes.shape[0]})"
                      for b in self._fifo[-count:]))

    def __next__(self) -> ExtractorBatch | ExtractSignal:
        """ Get the next batch of data from the iterator. Depending on the plugin type calling this
        iterator, a batch object will be returned for the given batch size of frames (for detect
        plugins) or faces (for all other plugins)

        Returns
        -------
        ExtractorBatch or ExtractSignal
            A new ExtractorBatch object containing the batch to process through the plugin or an
            ExtractSignal

        Raises
        ------
        StopIteration
            When the input is exhausted
        """
        flush = self._handle_signals()
        if flush:
            return flush

        while True:
            self._check_error()
            retval = self._from_fifo()  # In loop as re-batching may need to run multiple times
            if retval is not None:
                return retval

            batch = self._from_queue()
            if batch is None:
                continue

            if isinstance(batch, ExtractorBatch) and batch.passthrough and self._fifo:
                raise RuntimeError("Pipeline must be empty when adding a passthrough object")

            if isinstance(batch, ExtractorBatch) and batch.passthrough:
                return batch

            if isinstance(batch, ExtractorBatch):
                self._rebatch_data(batch)
                continue

            return self._handle_inbound_signal(batch)


class InterimIterator(ExtractIterator[ExtractorBatch, ExtractorBatch]):
    """ An iterator that simply collects interim ExtractorBatch objects from the given queue and
    yields them

    Parameters
    ----------
    queue
        The inbound queue to the plugin
    name
        The plugin name and process calling this iterator
    plugin_type
        The type of extractor plugin that this iterator is serving
    batch_size
        The batch size that data should be returned from the iterator
    """
    def __next__(self) -> ExtractorBatch | ExtractSignal:
        """ Get the next batch of data from the iterator

        Returns
        -------
        ExtractorBatch or ExtractSignal
            A new ExtractorBatch object containing the batch to process through the plugin or an
            ExtractSignal

        Raises
        ------
        StopIteration
            When the input is exhausted
        """
        batch = ExtractSignal.SHUTDOWN
        while True:
            self._check_error()
            batch = self._from_queue()
            if batch is not None:
                break

        if batch == ExtractSignal.SHUTDOWN:
            logger.debug("[%s] EOF Received", self._name)
            raise StopIteration

        if batch == ExtractSignal.FLUSH:
            logger.debug("[%s] FLUSH Received", self._name)

        logger.trace("[%s] Releasing batch: %s",  # type:ignore[attr-defined]
                     self._name, batch.name if isinstance(batch, ExtractSignal) else batch)
        return batch


class OutputIterator(ExtractIterator[ExtractorBatch, ExtractMedia]):
    """ Handles parsing incoming ExtractorBatch objects into ExtractMedia objects and yielding one
    frame at a time from the pipeline

    Parameters
    ----------
    queue
        The output queue from the plugin runner
    name
        The plugin name and process calling this iterator
    plugin_type
        The type of extractor plugin that this iterator is serving
    batch_size
        The batch size that data should be returned from the iterator
    """
    def _to_detected_faces(self, batch: ExtractorBatch, start: int, length: int
                           ) -> list[DetectedFace]:
        """Split the given batch object into detected faces across the given start and length

        Parameters
        ----------
        batch
            The ExtractorBatch containing the data to create DetectedFace objects from
        start
            The starting index to start collecting faces from
        length
            The number of faces to collect from the starting index

        Returns
        -------
        The list of detected faces at the given locations
        """
        return [
            DetectedFace(
                left=int(box[0]),
                top=int(box[1]),
                width=int(box[2] - box[0]),
                height=int(box[3] - box[1]),
                landmarks_xy=None if batch.landmarks is None else batch.landmarks[start + idx],
                mask={k: Mask(storage_size=v.storage_size,
                              storage_centering=v.centering).add(v.masks[start + idx],
                                                                 v.matrices[start + idx])
                      for k, v in batch.masks.items()},
                identity={k: v[start + idx] for k, v in batch.identities.items()}
                )
            for idx, box in enumerate(batch.bboxes[start:start + length])
            ]

    def _to_extract_media(self, batch: ExtractorBatch) -> None:
        """ Process the incoming batch data into ExtractMedia objects and return the next stored in
        local cache for output

        Parameters
        ----------
        batch
            The output ExtractorBatch object from a plugin
        """
        merge = self._fifo and batch.filenames[0] == self._fifo[-1].filename
        lengths = batch.lengths
        starts = np.cumsum(lengths, dtype="int32") - lengths
        for idx, (filename, image, source, start, length) in enumerate(zip(batch.filenames,
                                                                           batch.images,
                                                                           batch.sources,
                                                                           starts,
                                                                           lengths)):
            faces = self._to_detected_faces(batch, start, length)
            if merge and idx == 0:
                logger.trace(  # type:ignore[attr-defined]
                    "[%s] Merging %s faces to last batch: '%s'",
                    self._name, len(faces), filename)
                queued = self._fifo[-1]
                queued.detected_faces.extend(faces)
                if not np.any(queued.image) and np.any(image):
                    # Image has been removed on previous extractor batch filter
                    queued.add_image(image)
            else:
                self._fifo.append(ExtractMedia(filename=filename,
                                               image=image,
                                               source=source,
                                               detected_faces=faces))
                if batch.frame_metadata is not None:
                    self._fifo[-1].add_frame_metadata(batch.frame_metadata[idx])

            logger.trace(  # type:ignore[attr-defined]
                "[%s] Split to ExtractMedia: %s (%s faces)",
                self._name,
                repr(self._fifo[-1].filename),
                len(self._fifo[-1].detected_faces))

    def _handle_passthrough_batch(self, batch: ExtractorBatch) -> ExtractMedia:
        """Handle a batch when it is a passthrough object

        Parameters
        ----------
        batch
            The batch that contains the passthrough object

        Returns
        -------
        The ExtractMedia object derived from the incoming ExtractorBatch

        Raises
        ------
        RuntimeError
            If there are items to be queued out of the FIFO
        ValueError
            If the batch does not contain exactly one frame
        """
        if self._fifo:
            raise RuntimeError("Pipeline must be empty when adding a passthrough object")
        if len(batch.filenames) != 1:
            raise ValueError("Exactly 1 image should exist when passing through")
        return ExtractMedia(batch.filenames[0],
                            batch.images[0],
                            self._to_detected_faces(batch, 0, batch.bboxes.shape[0]),
                            batch.sources[0],
                            batch.is_aligned,
                            batch.passthrough)

    def __next__(self) -> ExtractMedia:
        """ Get the next batch of data from the iterator

        Returns
        -------
        ExtractMedia
            An ExtractMedia object for a single frame

        Raises
        ------
        StopIteration
            When the input is exhausted
        """
        self._handle_signals()
        while True:
            self._check_error()
            retval = self._from_fifo()
            if retval is not None:
                return retval

            batch = self._from_queue()
            if batch is None:
                continue

            if isinstance(batch, ExtractSignal):
                batch = self._handle_inbound_signal(batch)
            if isinstance(batch, ExtractMedia):
                return batch
            if batch == ExtractSignal.FLUSH:
                continue  # Don't flush to output. Wait for next batch

            assert isinstance(batch, ExtractorBatch)
            if batch.passthrough:
                return self._handle_passthrough_batch(batch)

            self._to_extract_media(batch)


__all__ = get_module_objects(__name__)

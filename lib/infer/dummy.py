#! /usr/env/bin/python3
"""Dummy runner for extract pipeline"""
from __future__ import annotations

import logging
import typing as T
from queue import Empty as QueueEmpty

from lib.utils import get_module_objects

from .objects import ExtractSignal
from .runner import ExtractRunner

if T.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from lib.align.alignments import PNGHeaderSourceDict
    from lib.align.detected_face import DetectedFace
    from .objects import ExtractMedia, ExtractorBatch
    
logger = logging.getLogger(__name__)


class DummyRunner(ExtractRunner):
    """A pseudo runner that matches interfaces of a standard ExtractRunner for passing through
    data when the pipeline is driven entirely by an alignments file (ie: no plugins are being
    loaded)

    Parameters
    ----------
    plugin
        The name of the plugin that this runner is to use
    """
    def __init__(self, plugin: str | None = None) -> None:
        plugin = "alignments" if plugin is None else plugin
        super().__init__(plugin)

    def pre_process(self) -> None:
        """overridden for Abstract Base Class but unused"""
        return

    def post_process(self) -> None:
        """overridden for Abstract Base Class but unused"""
        return

    # Hideously C+P'd overload just so typechecker doesn't complain
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
    def put(self,  # pylint:disable=arguments-differ
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
        """Put a frame into the dummy runner.

        Parameters
        ----------
        filename
            The filename of the frame
        image
            The loaded frame as UINT8 BGR array
        source
            The full path to the source folder or video file. Default: ``None`` (Not provided)
        detected_faces
            The detected face objects for the frame. ``None`` if not any. This cannot be ``None``
            for dummy runners. Default: ``None``
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

        Raises
        ------
        AssertionError
            If no detected faces are provided
        """
        assert detected_faces is not None
        super().put(filename,
                    image,
                    detected_faces=detected_faces,
                    source=source,
                    is_aligned=is_aligned,
                    frame_metadata=frame_metadata,
                    passthrough=passthrough)

    def put_direct(self,
                   filename: str,
                   image: npt.NDArray[np.uint8],
                   detected_faces: list[DetectedFace],
                   is_aligned: bool = False,
                   frame_size: tuple[int, int] | None = None) -> ExtractorBatch:
        # This makes no sense for a dummy runner so raise not implemented
        raise NotImplementedError

    def __next__(self) -> ExtractMedia:
        """Obtain the next item from the plugin's output

        Returns
        -------
        The media object with populated detected faces for a frame
        """
        while True:
            try:
                retval = self._queues["out"].get(timeout=1)
            except QueueEmpty:
                logger.trace("[%s] No item available",  # type:ignore[attr-defined]
                             self.__class__.__name__)
                continue
            if isinstance(retval, ExtractMedia):
                return retval
            if retval == ExtractSignal.SHUTDOWN:
                raise StopIteration
            if retval == ExtractSignal.FLUSH:
                continue  # Wait for next job


__all__ = get_module_objects(__name__)

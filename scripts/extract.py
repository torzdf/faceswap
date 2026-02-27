#!/usr/bin python3
""" Main entry point to the extract process of FaceSwap """
from __future__ import annotations

import logging
import os
import sys
import typing as T
from time import sleep

from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

from lib.align.aligned_utils import get_centered_size
from lib.align.detected_face import DetectedFace
from lib.infer import Detect, Align, Identity, Mask
from lib.infer.identity import FilterLoader
from lib.infer.runner import DummyRunner
from lib.infer.objects import ExtractMedia
from lib.image import (encode_image, generate_thumbnail, ImagesLoader, ImagesSaver)
from lib.logger import parse_class_init
from lib.multithreading import ErrorState, FSThread
from lib.utils import (get_folder, get_module_objects, handle_deprecated_cli_opts,
                       IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)

from .fsmedia import Alignments, finalize

if T.TYPE_CHECKING:
    from argparse import Namespace
    from lib.align.aligned_face import AlignedFace
    from lib.align.alignments import (AlignmentDict, AlignmentFileDict, PNGHeaderDict)
    from lib.infer.runner import ExtractRunner

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """ Holds information about each input batch being processed through extract

    Parameters
    ----------
    loader
        The images loader for the batch
    alignments
        The alignments for the input
    """
    loader: Loader
    """The images loader for the batch"""
    alignments: Alignments
    """The alignments for the input"""


class Extract:
    """ The Faceswap Face Extraction Process.

    The extraction process is responsible for detecting faces in a series of images/video, aligning
    them and optionally collecting further data about each face leveraging various user selected
    plugins

    Parameters
    ----------
    arguments
        The arguments to be passed to the extraction process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        args = handle_deprecated_cli_opts(arguments,
                                          additional={"K": ("to skip saving faces", True, None)})
        args = self._validate_compatible_args(args)
        self._face_filter = FilterLoader(args.ref_threshold, args.filter, args.nfilter)
        self._pipeline = self._load_pipeline(args)
        input_locations = self._get_input_locations(args.input_dir, args.batch_mode)
        self._validate_batch_mode(args.batch_mode, input_locations, args)

        file_input = args.detector == "file" or args.aligner == "file"
        save_alignments = self._should_save_alignments(args)
        self._batches = [BatchInfo(ld := Loader(self._pipeline,
                                                input_location,
                                                file_input,
                                                args.extract_every_n,
                                                args.skip_existing,
                                                args.skip_faces,
                                                idx == len(input_locations) - 1),
                                   Alignments(args.alignments_path,
                                              ld.location,
                                              is_extract=True,
                                              skip_existing_frames=args.skip_existing,
                                              skip_existing_faces=arguments.skip_faces,
                                              plugin_is_file=file_input,
                                              save_alignments=save_alignments,
                                              input_is_video=ld.is_video))
                         for idx, input_location in enumerate(input_locations)]

        self._output = Output(self._pipeline,
                              args.output_dir,
                              args.size,
                              args.min_scale,
                              self._batches,
                              args.save_interval,
                              args.debug_landmarks)

    @classmethod
    def _get_input_locations(cls, input_location: str, batch_mode: bool) -> list[str]:
        """ Obtain the full path to input locations. Will be a list of locations if batch mode is
        selected, or a list containing a single location if batch mode is not selected.

        Parameters
        ----------
        input_location
            The full path to the input location. Either a video file, a folder of images or a
            folder containing either/or videos and sub-folders of images (if batch mode is
            selected)
        batch_mode
            ``True`` if extract is running in batch mode

        Returns
        -------
        list
            The list of input location paths
        """
        if not batch_mode:
            return [input_location]

        if os.path.isfile(input_location):
            logger.warning("Batch mode selected but input is not a folder. Switching to normal "
                           "mode")
            return [input_location]

        retval = [os.path.join(input_location, fname)
                  for fname in os.listdir(input_location)
                  if (os.path.isdir(os.path.join(input_location, fname))  # folder images
                      and any(os.path.splitext(iname)[-1].lower() in IMAGE_EXTENSIONS
                              for iname in os.listdir(os.path.join(input_location, fname))))
                  or os.path.splitext(fname)[-1].lower() in VIDEO_EXTENSIONS]  # video

        retval = list(sorted(retval))
        logger.debug("[Extract] Input locations: %s", retval)
        return retval

    @classmethod
    def _validate_compatible_args(cls, args: Namespace) -> Namespace:
        """Some cli arguments are not compatible with each other. If conflicting arguments have
        been selected, log a warning and make necessary changes

        Parameters
        ----------
        args
            The command line arguments to be checked and updated for conflicts

        Returns
        -------
        The updated command line arguments
        """
        # Can't run a detector if importing landmarks
        if args.aligner == "file" and args.detector != "file":
            logger.warning("Detecting faces is not compatible with importing landmarks from a "
                           "file. Setting Detector to 'file'")
            args.detector = "file"
        # Impossible to skip existing when not running detection
        if args.skip_existing and args.detector == "file":
            logger.warning("Skipping existing frames is not compatible with importing from a file "
                           "for detection. Disabling 'skip_existing'")
            args.skip_existing = False
        # Impossible to get missing faces when we do not have a detector or aligner
        if args.skip_faces and (args.detector == "file" or args.aligner == "file"):
            logger.warning("Skipping existing faces is not compatible with importing from a file. "
                           "Disabling 'skip_existing_faces'")
            args.skip_faces = False
        # Face filtering needs a recognition plugin
        if (args.filter or args.nfilter) and not args.identity:
            logger.warning("Face-filtering is enabled, but an identity plugin has not been "
                           "selected. Selecting 'T-Face' plugin")
            args.identity = ["t-face"]
        # We can only use 1 identity for face filtering, so we select the first given
        if (args.filter or args.nfilter) and len(args.identity) > 1:
            logger.warning("Face-filtering is enabled, but multiple identity plugins have been "
                           "selected. Using '%s' for filtering", args.identity[0])
        return args

    def _validate_batch_mode(self, batch_mode: bool,
                             input_locations: list[str],
                             args: Namespace) -> None:
        """ Validate the command line arguments.

        If batch-mode selected and there is only one object to extract from, then batch mode is
        disabled

        If processing in batch mode, some of the given arguments may not make sense, in which case
        a warning is shown and those options are reset.

        Parameters
        ----------
        batch_mode
            ``True`` if extract is running in batch mode
        input_locations
            The discovered input locations within the input folder
        args
            The passed in command line arguments that may require amending
        """
        if not batch_mode:
            return

        if not input_locations:
            logger.error("Batch mode selected, but no valid files found in input location: '%s'. "
                         "Exiting.", args.input_dir)
            sys.exit(1)

        if args.alignments_path:
            logger.warning("Custom alignments path not supported for batch mode. "
                           "Reverting to default.")
            args.alignments_path = None

    def _should_save_alignments(self, arguments: Namespace) -> bool:
        """ Decide whether alignments should be saved from the given command line arguments and
        output suitable information

        Parameters
        ---------
        arguments
            The arguments generated from Faceswap's command line arguments

        Returns
        -------
        ``True`` if alignments should be saved
        """
        if arguments.detector == arguments.aligner == "file" and (
                arguments.masker is None and arguments.identity is None):
            logger.debug("[Extract] Extracting directly from file. Not saving alignments")
            return False
        if arguments.detector == arguments.aligner == "file" and arguments.extract_every_n > 1:
            logger.warning("Alignments loaded from file, EEN > 1 and additional plugins selected.")
            logger.warning("The extracted faces will contain the additional plugin data, but an "
                           "updated Alignments File will not be saved.")
            return False
        if arguments.detector == arguments.aligner == "file":
            logger.info("Alignments file will be updated with data from additional plugins")
        return True

    def _load_pipeline(self, arguments: Namespace) -> ExtractRunner:  # noqa[C901]
        """ Create the extraction pipeline

        Parameters
        ---------
        arguments
            The arguments generated from Faceswap's command line arguments

        Returns
        -------
        ExtractRunner or None
            The final runner, with input interfaces, from the pipeline or ``None`` if the input and
            output is being driven directly from the alignments file
        """
        retval = None
        conf_file = arguments.configfile
        try:
            if arguments.detector != "file":
                retval = Detect(arguments.detector,
                                rotation=arguments.rotate_images,
                                min_size=arguments.min_size,
                                max_size=arguments.max_size,
                                config_file=conf_file)(retval)
            if arguments.aligner != "file":
                retval = Align(arguments.aligner,
                               re_feeds=arguments.re_feed,
                               re_align=arguments.re_align,
                               normalization=arguments.normalization,
                               filters=arguments.align_filters,
                               config_file=conf_file)(retval)
            if arguments.masker is not None:
                for masker in arguments.masker:
                    retval = Mask(masker, config_file=conf_file)(retval)
            if arguments.identity:
                for idx, identity in enumerate(arguments.identity):
                    retval = Identity(identity,
                                      self._face_filter.threshold,
                                      config_file=conf_file)(retval)
                    if self._face_filter.enabled and idx == 0:
                        # Add the first selected identity plugin
                        self._face_filter.add_identity_plugin(retval)
            retval = DummyRunner()() if retval is None else retval
        except Exception:
            logger.debug("[Extract] Error during pipeline initialization")
            if retval is not None:
                retval.stop()
            raise
        logger.debug("[Extract] Pipeline output: %s", retval)
        return retval

    def process(self) -> None:
        """ Run the extraction process """
        try:
            if self._face_filter.enabled:
                self._face_filter.get_embeddings(self._pipeline)
            self._output.start()
            for batch in self._batches:
                batch.loader.start(batch.alignments.data)
                batch.loader.join()
                if ErrorState.has_error():
                    ErrorState.check_and_raise()
            self._output.join()
        except Exception:
            self._pipeline.stop()
            raise


class Loader:  # pylint:disable=too-many-instance-attributes
    """ Loads images/video frames from disks and puts to queue for feeding the extraction pipeline

    Parameters
    ----------
    pipeline
        The final plugin in the extraction pipeline
    input_path
        Full path to a folder of images or a video file
    input_is_file
        ``True`` if the input plugin to the pipeline is an alignments file (fsa or json) so
        detected faces should be loaded from the file and passed into the pipeline
    extract_every
        The number of frames to extract from the source. 1 will extract every frame, 5 every 5th
        frame etc
    skip_existing_frames
        ``True`` if existing extracted frames should be skipped
    skip_existing_faces
        ``True`` if frames with existing face detections should be skipped
    is_final
        ``True`` if this loader is for the final batch being processed
    """
    def __init__(self,
                 pipeline: ExtractRunner,
                 input_path: str,
                 input_is_file: bool,
                 extract_every: int,
                 skip_existing_frames: bool,
                 skip_existing_faces: bool,
                 is_final: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self.location = input_path
        """Full path to the input location for the loader"""
        self.existing_count = 0
        """The number of frames that pre-exist within the alignments file that will be skipped
        because skip_existing/skip_existing_faces has been selected"""

        self._input_is_file = input_is_file
        self._pipeline = pipeline
        self._is_final = is_final
        self._extract_every = extract_every
        self._skip_frames = skip_existing_frames
        self._skip_faces = skip_existing_faces

        self._images = ImagesLoader(input_path)
        self._thread = FSThread(self._load, name="ExtractLoader")
        self._alignments: dict[str, AlignmentDict] = {}
        self._missing_count = 0
        self._seen: set[str] = set()
        self._ready = False

    @property
    def count(self) -> int:
        """The number of frames to be processed"""
        # Wait until skip list has been processed before allowing another thread to call the count
        while True:
            if self._ready:
                break
            sleep(0.25)
            continue
        return self._images.process_count

    @property
    def is_video(self) -> bool:
        """``True`` if the input location is a video file, ``False`` for folder of images"""
        return self._images.is_video

    def _set_skip_list(self) -> None:
        """ Add the skip list to the image loader

        Checks against `extract_every_n` and the existence of alignments data (can exist if
        `skip_existing` or `skip_existing_faces` has been provided) and compiles a list of frame
        indices that should not be processed, providing these to :class:`lib.image.ImagesLoader`.
        """
        existing = list(self._alignments)
        if self._extract_every == 1 and not existing:
            logger.debug("[Extract.Loader] No frames to be skipped")
            self._ready = True
            return

        skip_een = set(i for i in range(self._images.count) if i % self._extract_every != 0)

        file_names = ([os.path.basename(f) for f in self._images.file_list]
                      if self._skip_frames or self._skip_faces else [])
        skip_frames = set(i for i, f in enumerate(file_names)
                          if f in existing) if self._skip_frames else set()
        skip_faces = (set(i for i, f in enumerate(file_names)
                          if self._alignments.get(f, {}).get("faces"))
                      if self._skip_faces else set())
        skip_exist = skip_frames.union(skip_faces)

        if self._extract_every > 1:
            logger.info("Skipping %s frames of %s for extract every %s",
                        len(skip_een), self._images.count, self._extract_every)
        if skip_exist:
            self.existing_count = len(skip_exist.difference(skip_een))
            logger.info("Skipping %s frames of %s for skip existing frames/faces",
                        self.existing_count, self._images.count - len(skip_een))

        skip = list(skip_exist.union(skip_een))
        logger.debug("[Extract.Loader] Total skip count: %s", len(skip))
        self._images.add_skip_list(skip)
        self._ready = True

    def _get_detected_faces(self, file_path: str) -> list[DetectedFace] | None:
        """When importing data, obtain the existing detected face objects for passing through the
        pipeline

        Parameters
        ----------
        file_path
            The full path to the image being loaded

        Returns
        -------
        list[DetectedFace] | None
            The imported detected face objects or ``None`` if data is not being imported
        """
        if not self._input_is_file:
            return None
        fname = os.path.basename(file_path)
        self._seen.add(fname)
        if fname not in self._alignments:
            self._missing_count += 1
            logger.verbose(  # type:ignore[attr-defined]
                "Adding frame with no detections as does not exist in import file: '%s'", fname)
            return []
        retval = [DetectedFace().from_alignment(a)
                  for a in self._alignments[fname].get("faces", [])]
        logger.trace(  # type:ignore[attr-defined]
            "[Extract.Loader] importing %s faces for file '%s'", len(retval), fname)
        return retval

    def _finalize(self) -> None:
        """Actions to run when the loader is exhausted"""
        if self._is_final:
            self._pipeline.stop()
        if self._missing_count > 0:
            logger.warning("%s images did not exist in the import file. Run in verbose mode to "
                           "see which files have been added with no detected faces.",
                           self._missing_count)
        processed_files = set(self._images.processed_file_list)
        if self._input_is_file and len(self._seen) != len(processed_files):
            logger.warning("%s images exist in the import file but do not exist on disk. Run in "
                           "verbose mode to see which files are missing.",
                           len(processed_files) - len(self._seen))
            logger.verbose(  # type:ignore[attr-defined]
                "Files in import file that do not exist on disk: %s",
                list(sorted(processed_files.difference(self._seen))))

    def _load(self) -> None:
        """ Load images from disk and pass to a queue for the extraction pipeline """
        logger.debug("[Extract.Loader] start")
        for filename, image in self._images.load():
            if ErrorState.has_error():
                ErrorState.check_and_raise()
            faces = self._get_detected_faces(filename)
            self._pipeline.put(filename, image, source=self.location, detected_faces=faces)
        self._finalize()
        logger.debug("[Extract.Loader] end")

    def start(self, alignments: dict[str, AlignmentDict]) -> None:
        """ Set the skip list and start loading images from disk

        Parameters
        ----------
        alignments
            Dictionary of existing alignments data for use when importing or skipping existing data
        """
        self._alignments = alignments
        self._set_skip_list()
        logger.debug("[Extract.Loader] start thread")
        self._thread.start()

    def join(self) -> None:
        """ Join the image loading thread """
        logger.debug("[Extract.Loader] join thread")
        self._thread.join()
        logger.debug("[Extract.Loader] joined thread")


class DebugLandmarks():
    """Draw debug landmarks on face output.

    Parameters
    ----------
    size
        The size of the extracted face image
    """
    def __init__(self, size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._size = size
        self._face_size = get_centered_size("head", "face", size)
        self._legacy_size = get_centered_size("head", "legacy", size)
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = size / 512
        self._font_pad = size // 64

    def _border_text(self,
                     image: np.ndarray,
                     text: str,
                     color: tuple[int, int, int],
                     position: tuple[int, int]) -> None:
        """Create text on an image with a black border

        Parameters
        ----------
        image
            The image to put bordered text on to
        text
            The text to place the image
        color
            The color of the text
        position
            The (x, y) co-ordinates to place the text
        """
        thickness = 2
        for idx in range(2):
            text_color = (0, 0, 0) if idx == 0 else color
            cv2.putText(image,
                        text,
                        position,
                        self._font,
                        self._font_scale,
                        text_color,
                        thickness,
                        lineType=cv2.LINE_AA)
            thickness //= 2

    def _annotate_face_box(self, face: AlignedFace) -> None:
        """Annotate the face extract box and print the original size in pixels

        Parameters
        ----------
        face
            The object containing the aligned face to annotate
        """
        assert face.face is not None
        color = (0, 255, 0)
        roi = face.get_cropped_roi(self._size, self._face_size, "face")
        cv2.rectangle(face.face, tuple(roi[:2]), tuple(roi[2:]), color, 1)

        # Size in top right corner
        roi_points = np.array([[roi[0], roi[1]],
                               [roi[0], roi[3]],
                               [roi[2], roi[3]],
                               [roi[2], roi[1]]])
        orig_roi = face.transform_points(roi_points, invert=True)
        size = int(round(((orig_roi[1][0] - orig_roi[0][0]) ** 2 +
                          (orig_roi[1][1] - orig_roi[0][1]) ** 2) ** 0.5))
        text_img = face.face.copy()
        text = f"{size}px"
        text_size = cv2.getTextSize(text, self._font, self._font_scale, 1)[0]
        pos_x = roi[2] - (text_size[0] + self._font_pad)
        pos_y = roi[1] + text_size[1] + self._font_pad

        self._border_text(text_img, text, color, (pos_x, pos_y))
        cv2.addWeighted(text_img, 0.75, face.face, 0.25, 0, face.face)

    def _print_stats(self, face: AlignedFace) -> None:
        """Print various metrics on the output face images

        Parameters
        ----------
        face
            The object containing the aligned face to annotate
        """
        assert face.face is not None
        text_image = face.face.copy()
        texts = [f"pitch: {face.pose.pitch:.2f}",
                 f"yaw: {face.pose.yaw:.2f}",
                 f"roll: {face.pose.roll: .2f}",
                 f"distance: {face.average_distance:.2f}"]
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255)]
        text_sizes = [cv2.getTextSize(text, self._font, self._font_scale, 1)[0] for text in texts]

        final_y = self._size - text_sizes[-1][1]
        pos_y = [(size[1] + self._font_pad) * (idx + 1)
                 for idx, size in enumerate(text_sizes)][:-1] + [final_y]
        pos_x = self._font_pad

        for idx, text in enumerate(texts):
            self._border_text(text_image, text, colors[idx], (pos_x, pos_y[idx]))

        # Apply text to face
        cv2.addWeighted(text_image, 0.75, face.face, 0.25, 0, face.face)

    def __call__(self, face: AlignedFace) -> None:
        """Draw debug annotations on an extracted face image

        Parameters
        ----------
        face
            The object containing the aligned face to annotate
        """
        assert face.face is not None
        # Landmarks
        for (pos_x, pos_y) in face.landmarks.astype("int32"):
            cv2.circle(face.face, (pos_x, pos_y), 1, (0, 255, 255), -1)
        # Pose
        center = (self._size // 2, self._size // 2)
        points = (face.pose.xyz_2d * self._size).astype("int32")
        cv2.line(face.face, center, tuple(points[1]), (0, 255, 0), 1)
        cv2.line(face.face, center, tuple(points[0]), (255, 0, 0), 1)
        cv2.line(face.face, center, tuple(points[2]), (0, 0, 255), 1)
        # Face centering
        self._annotate_face_box(face)
        # Legacy centering
        roi = face.get_cropped_roi(self._size, self._legacy_size, "legacy")
        cv2.rectangle(face.face, tuple(roi[:2]), tuple(roi[2:]), (0, 0, 255), 1)
        self._print_stats(face)


class Output:  # pylint:disable=too-many-instance-attributes
    """ Handles output processing and saving of extracted faces

    Parameters
    ----------
    pipeline
        The output runner from the extraction pipeline
    output_folder
        The full path to the output folder to save extracted faces. ``None`` to not save faces
    size
        The size to save extracted faces at
    min_scale
        The minimum percentage of the output size that should be accepted for outputting a face
        to disk
    batches
        The information about each batch that is to be processed
    save_interval
        How often to save the alignments file
    debug_landmarks
        ``True`` to annotate the output images with debug data
    """
    def __init__(self,
                 pipeline: ExtractRunner,
                 output_folder: str | None,
                 size: int,
                 min_scale: int,
                 batches: list[BatchInfo],
                 save_interval: int,
                 debug_landmarks: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._pipeline = pipeline
        self._size = size
        self._batches = batches
        self._save_interval = save_interval
        self._min_size = self._get_min_size(size, min_scale)
        self._saver: None | ImagesSaver = None
        self._outputs = self._get_outputs(output_folder)
        self._thread = FSThread(self._process, name="ExtractOutput")
        self._debug = DebugLandmarks(size) if debug_landmarks else None
        self._verify_output = False
        self._faces_count = 0
        self._scale_skip_count = 0

    @classmethod
    def _get_min_size(cls, extract_size: int, min_scale: int) -> int:
        """ Obtain the minimum size that a face has been resized from to be included as a valid
        extract.

        Parameters
        ----------
        extract_size
            The requested size of the extracted images
        min_scale
            The percentage amount that has been supplied for valid faces (as a percentage of
            extract size)

        Returns
        -------
        The minimum size, in pixels, that a face is resized from to be considered valid
        """
        retval = 0 if min_scale == 0 else max(4, int(extract_size * (min_scale / 100.)))
        logger.debug("[Extract.Output] Extract size: %s, min percentage size: %s, min_size: %s",
                     extract_size, min_scale, retval)
        return retval

    def _get_outputs(self, output_folder: str | None) -> list[str | None]:
        """ Obtain the locations to save the output for each batch input location

        Parameters
        ----------
        output_folder
            The full path to the output folder to save extracted faces. ``None`` to not save faces

        Returns
        -------
        The output locations for each input batch. ``None`` if faces are not to be saved
        """
        num_batches = len(self._batches)
        retval: list[str | None]
        if not output_folder:
            logger.debug("[Extract.Output] No save location selected")
            return [None for _ in range(num_batches)]
        out_folder = get_folder(output_folder)
        if num_batches == 1:
            logger.debug("[Extract.Output] Single save location: '%s'", out_folder)
            return [out_folder]
        retval = [os.path.join(out_folder,
                               os.path.splitext(os.path.basename(b.loader.location))[0])
                  for b in self._batches]
        logger.debug("[Extract.Output] Save locations: %s", retval)
        return retval

    def _should_output(self, face: AlignedFace) -> bool:
        """Test whether the face should be saved based on the given minimum scale option

        Parameters
        ----------
        face
            The aligned face to check for scaling

        Returns
        -------
        bool
            ``True`` if the face should be output. ``False`` if it falls below the minimum scale
        """
        if self._min_size <= 0:
            return True
        roi = face.original_roi
        tl, tr = roi[0], roi[3]
        len_x = tr[0] - tl[0]
        len_y = tr[1] - tl[1]
        size = len_y if tl[1] == tr[1] else int(((len_x ** 2) + (len_y ** 2)) ** 0.5)
        if size < self._min_size:
            self._scale_skip_count += 1
            return False
        return True

    def _save_face(self,
                   face: DetectedFace,
                   filename: str,
                   face_index: int,
                   frame_size: tuple[int, int],
                   alignments_version: float,
                   is_video: bool) -> None:
        """ Encode the face with PNG Header information and save to disk

        Parameters
        ----------
        face
            The detected face object containing the aligned face to save
        filename
            The original filename (basename) of the frame that contains the face
        face_index
            The index of the face within the frame
        frame_size
            The (height, width) of the original frame
        alignments_version
            The current alignments file version
        is_video
            ``True`` if the input is a video otherwise ``False``
        """
        if self._saver is None:
            return
        if not self._should_output(face.aligned):
            return
        filename = f"{os.path.splitext(filename)[0]}_{face_index}.png"
        meta: PNGHeaderDict = {
            "alignments": face.to_png_meta(),
            "source": {"alignments_version": alignments_version,
                       "original_filename": filename,
                       "face_index": face_index,
                       "source_filename": filename,
                       "source_is_video": is_video,
                       "source_frame_dims": frame_size}}
        assert face.aligned.face is not None
        img = encode_image(face.aligned.face, ".png", metadata=meta)
        self._saver.save(filename, img)

    def _process_faces(self, media: ExtractMedia, alignments: Alignments, is_video: bool) -> None:
        """ Process the detected face objects into aligned faces, generate the thumbnails and run
        any post process actions

        Parameters
        ----------
        media
            An extract media object for a frame from the extraction pipeline
        alignments
            The alignments object that is to contain these faces
        is_video
            ``True`` if the input is a video otherwise ``False``
        """
        faces: list[AlignmentFileDict] = []
        basename = os.path.basename(media.filename)
        for idx, face in enumerate(media.detected_faces):
            face.load_aligned(media.image, size=self._size, centering="head")
            face.thumbnail = generate_thumbnail(face.aligned.face, size=96, quality=60)
            if self._debug is not None:
                self._debug(face.aligned)
            self._save_face(face, basename, idx, media.image_size, alignments.version, is_video)
            faces.append(face.to_alignment())

        media.remove_image()
        alignments.data[basename] = {"faces": faces, "video_meta": {}}
        faces_count = len(media.detected_faces)
        if faces_count == 0:
            logger.verbose("No faces were detected in image: %s",  # type: ignore
                           os.path.basename(media.filename))

        if not self._verify_output and faces_count > 1:
            self._verify_output = True
        self._faces_count += faces_count

    def _set_saver(self, output: str | None) -> None:
        """Close the currently active saver and set the next :attr:`_saver` for the given output

        Parameters
        ----------
        output
            The full path to the next output location
        """
        if self._saver is not None:
            self._saver.close()
        if output is None:
            self._saver = None
        else:
            self._saver = ImagesSaver(get_folder(output), as_bytes=True)
        logger.debug("[Extract.Output] Set image saver to location: %s",
                     repr(self._saver if self._saver is None else self._saver.location))

    def _finalize_batch(self, batch: BatchInfo, batch_index: int) -> None:
        """ Actions to perform when an input batch has finished processing.

        Parameters
        ----------
        batch
            The information about the batch that has finished processing
        batch_index
            The index of the batch in :attr:`_self._batches`
        """
        logger.debug("[Extract.Output] Finalizing batch: %s", batch)
        if batch.alignments.save_alignments:
            if not self._save_interval:
                batch.alignments.backup()
            batch.alignments.save()
        count = batch.loader.count - batch.loader.existing_count
        if self._scale_skip_count > 0:
            logger.info("%s faces not output as they are below the minimum size of %spx",
                        self._scale_skip_count, self._min_size)
            logger.info("These faces still exist within the alignments file")
        finalize(count, self._faces_count, self._verify_output)
        self._verify_output = False
        output = None if batch_index == len(self._outputs) - 1 else self._outputs[batch_index + 1]
        self._set_saver(output)
        self._faces_count = 0
        self._scale_skip_count = 0
        del batch.alignments

    def _process(self) -> None:
        """ Process the output from the extraction pipeline within a thread """
        logger.debug("[Extract.Output] start")
        total_batches = len(self._batches)
        self._set_saver(self._outputs[0])
        if self._saver is not None and self._min_size > 0:
            logger.info("Only outputting faces that have been resized from a minimum resolution "
                        "of %spx", self._min_size)

        for batch_idx, batch in enumerate(self._batches):
            msg = f" job {batch_idx + 1} of {total_batches}" if total_batches > 1 else ""
            logger.info("Processing%s: '%s'", msg, batch.loader.location)
            if self._saver is not None:
                logger.info("Faces output: '%s'", self._saver.location)
            has_started = False
            save_interval = 0 if not batch.alignments.save_alignments else self._save_interval
            with tqdm(desc="Extracting faces",
                      total=batch.loader.count,
                      leave=True,
                      smoothing=0) as prog_bar:
                if batch_idx > 0:  # Update for batch picked up at end of previous batch
                    prog_bar.update(1)

                for idx, media in enumerate(self._pipeline):
                    if not has_started:
                        prog_bar.reset()  # Delay before first output, reset timer for better it/s
                        has_started = True

                    if media.source != batch.loader.location:
                        self._finalize_batch(batch, batch_idx)
                        next_batch = self._batches[batch_idx + 1]
                        self._process_faces(media, next_batch.alignments,
                                            next_batch.loader.is_video)
                        break

                    self._process_faces(media, batch.alignments, batch.loader.is_video)
                    if save_interval and (idx + 1) % save_interval == 0:
                        batch.alignments.save()
                    if prog_bar.n + 1 > prog_bar.total:
                        # Don't switch to unknown mode when frame count is under
                        prog_bar.total += 1
                    prog_bar.update(1)

        self._finalize_batch(self._batches[-1], len(self._batches) - 1)
        logger.debug("[Extract.Output] end")

    def start(self) -> None:
        """ Start the output thread """
        logger.debug("[Extract.Output] start thread")
        self._thread.start()

    def join(self) -> None:
        """ Join the output thread """
        logger.debug("[Extract.Output] join thread")
        self._thread.join()
        logger.debug("[Extract.Output] joined thread")


__all__ = get_module_objects(__name__)

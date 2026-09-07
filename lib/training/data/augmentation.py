#!/usr/bin/env python3
""" Perform random augmentation on batches of training images for FaceSwap models

Augmentation varies the training data by randomly altering images through color and contrast,
geometric transformation and warping. This helps the model generalise and reduce overfitting. The
behaviour is controlled by the settings in :file:`config.train.ini`
"""
from __future__ import annotations
import logging
import typing as T
from dataclasses import dataclass

import cv2
import numexpr as ne
import numpy as np
from scipy.interpolate import griddata

from lib.align.aligned_utils import batch_create_matrices
from lib.image import batch_convert_color
from lib.logger import format_array, parse_class_init
from lib.utils import get_module_objects
from plugins.train.trainer import trainer_config as cfg

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class ConstantsColor:
    """ Dataclass holding the constants used for color and contrast enhancement

    Parameters
    ----------
    clahe_base_contrast
        The base contrast value used when generating CLAHE grid sizes
    clahe_chance
        The probability of applying Contrast Limited Adaptive Histogram Equalization
    clahe_max_size
        The maximum tile size used when generating random CLAHE grids
    lab_adjust
        The adjustment amounts applied to the L*A*B color channels
    """
    clahe_base_contrast: int
    """ The probability of applying Contrast Limited Adaptive Histogram Equalization """
    clahe_chance: float
    """ The probability of applying Contrast Limited Adaptive Histogram Equalization """
    clahe_max_size: int
    """ The maximum tile size used when generating random CLAHE grids """
    lab_adjust: np.ndarray
    """ The adjustment amounts applied to the L*A*B color channels """


@dataclass
class ConstantsTransform:
    """ Dataclass holding the constants used for geometric transformations

    Parameters
    ----------
    rotation
        The rotation range applied to each image
    zoom
        The zoom range applied to each image
    shift
        The pixel shift range applied to each image
    flip
        The probability of flipping an image horizontally
    """
    rotation: int
    """ The rotation range applied to each image """
    zoom: float
    """ The zoom range applied to each image """
    shift: float
    """ The pixel shift range applied to each image """
    flip: float
    """ The probability of flipping an image horizontally """


@dataclass
class ConstantsWarp:
    """ Dataclass holding the constants used for image warping

    Parameters
    ----------
    maps
        The stacked (x, y) mappings used to warp each image
    pad
        The padding dimensions applied when warping images
    slices
        The slices used to extract a warped image from its mapped grid
    scale
        The scaling applied to the random variable driving standard warping
    lm_edge_anchors
        The edge anchors used for landmark based warping
    lm_grids
        The grids used for landmark based warping
    lm_scale
        The scaling applied to the random variable driving landmark based warping
    """
    maps: np.ndarray
    """ The stacked (x, y) mappings used to warp each image """
    pad: tuple[int, int]
    """ The padding dimensions applied when warping images """
    slices: slice
    """ The slices used to extract a warped image from its mapped grid """
    scale: float
    """ The scaling applied to the random variable driving standard warping """
    lm_edge_anchors: np.ndarray
    """ The edge anchors used for landmark based warping """
    lm_grids: np.ndarray
    """ The grids used for landmark based warping """
    lm_scale: float
    """ The scaling applied to the random variable driving landmark based warping """

    def __repr__(self) -> str:
        """ Prevent full size arrays in __repr__ """
        params = {k: format_array(v) if isinstance(v, np.ndarray) else f"{v!r}"
                  for k, v in self.__dict__.items()}
        str_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({str_params})"


@dataclass
class ConstantsAugmentation:
    """ Dataclass holding the constants that control every augmentation stage

    This aggregates the color and contrast, transformation and warping constants into a single
    object. Create instances with the :func:`from_config` classmethod rather than calling the
    constructor directly so they are derived from the user configuration

    Parameters
    ----------
    color
        The constants used for adjusting color and contrast in an image
    transform
        The constants used for geometric transformation of an image
    warp
        The constants used for warping an image

    Example
    -------
    >>> constants = ConstantsAugmentation.from_config(processing_size=256,
    ...                                               batch_size=16)
    """
    color: ConstantsColor
    """ The constants used for adjusting color and contrast in an image """
    transform: ConstantsTransform
    """ The constants used for geometric transformation of an image """
    warp: ConstantsWarp
    """ The constants used for warping an image """

    @classmethod
    def _get_clahe(cls, size: int) -> tuple[int, float, int]:
        """ Get the CLAHE constants from user config

        Parameters
        ----------
        size
            The size of image to augment the data for

        Returns
        -------
        clahe_base_contrast
            The base contrast value used when generating CLAHE grid sizes
        clahe_chance
            The probability of applying Contrast Limited Adaptive Histogram Equalization
        clahe_max_size
            The maximum tile size used when generating random CLAHE grids
        """
        clahe_base_contrast = max(2, size // 128)
        clahe_chance = cfg.Augmentation.color_clahe_chance() / 100
        clahe_max_size = cfg.Augmentation.color_clahe_max_size()
        logger.debug("[AugConstants] clahe_base_contrast: %s, clahe_chance: %s, "
                     "clahe_max_size: %s", clahe_base_contrast, clahe_chance, clahe_max_size)
        return clahe_base_contrast, clahe_chance, clahe_max_size

    @classmethod
    def _get_lab(cls) -> np.ndarray:
        """ Load the random L*A*B augmentation constants

        Returns
        -------
        The adjustment amounts applied to the L*A*B color channels
        """
        amount_l = cfg.Augmentation.color_lightness() / 100.
        amount_ab = cfg.Augmentation.color_ab() / 100.

        lab_adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")
        logger.debug("[AugConstants] lab_adjust: %s", lab_adjust)
        return lab_adjust

    @classmethod
    def _get_color(cls, size: int) -> ConstantsColor:
        """ Get the image enhancement constants from user config

        Parameters
        ----------
        size
            The size of image to augment the data for

        Returns
        -------
        The constants used for color and contrast enhancement
        """
        clahe_base_contrast, clahe_chance, clahe_max_size = cls._get_clahe(size)
        retval = ConstantsColor(clahe_base_contrast=clahe_base_contrast,
                                clahe_chance=clahe_chance,
                                clahe_max_size=clahe_max_size,
                                lab_adjust=cls._get_lab())
        logger.debug("[AugConstants] color: %s", retval)
        return retval

    @classmethod
    def _get_transform(cls, size: int) -> ConstantsTransform:
        """Load the random transform constants

        Parameters
        ----------
        size
            The size of image to augment the data for

        Returns
        -------
        The constants used for geometric transformation.
        """
        retval = ConstantsTransform(rotation=cfg.Augmentation.rotation_range(),
                                    zoom=cfg.Augmentation.zoom_amount() / 100.,
                                    shift=(cfg.Augmentation.shift_range() / 100.) * size,
                                    flip=cfg.Augmentation.flip_chance() / 100.)
        logger.debug("[AugConstants] transform: %s", retval)
        return retval

    @classmethod
    def _get_warp_to_landmarks(cls, size: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """ Load the warp-to-landmarks augmentation constants

        Parameters
        ----------
        size
            The size of image to augment the data for
        batch_size
            The batch size that augmented data is being prepared for

        Returns
        -------
        edge_anchors
            The edge anchors used for landmark based warping
        grids
            The grids used for landmark based warping
        """
        p_mx = size - 1
        p_hf = (size // 2) - 1
        edge_anchors = np.array([(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                                 (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]).astype("int32")
        edge_anchors = np.broadcast_to(edge_anchors, (batch_size, 8, 2))
        grids = np.mgrid[0: p_mx: complex(size),  # type:ignore[misc]  # pylint:disable=no-member
                         0: p_mx: complex(size)].astype("float32")  # type:ignore[misc]

        logger.debug("[AugConstants] edge_anchors: (%s, %s), grids: (%s, %s)",
                     edge_anchors.shape, edge_anchors.dtype,
                     grids.shape, grids.dtype)  # pylint:disable=no-member
        return edge_anchors, grids

    @classmethod
    def _get_warp(cls, size: int, batch_size: int) -> ConstantsWarp:
        """ Load the warp augmentation constants

        Parameters
        ----------
        size
            The size of image to augment the data for
        batch_size
            The batch size that augmented data is being prepared for

        Returns
        -------
        The constants used for image warping
        """
        lm_edge_anchors, lm_grids = cls._get_warp_to_landmarks(size, batch_size)

        warp_range = np.linspace(0, size, 5, dtype='float32')
        warp_map_x = np.broadcast_to(warp_range, (batch_size, 5, 5)).astype("float32")
        warp_map_y = np.broadcast_to(warp_map_x[0].T, (batch_size, 5, 5)).astype("float32")
        warp_pad = int(1.25 * size)

        retval = ConstantsWarp(maps=np.stack((warp_map_x, warp_map_y), axis=1),
                               pad=(warp_pad, warp_pad),
                               slices=slice(warp_pad // 10, -warp_pad // 10),
                               scale=5 / 256 * size,  # Normal random variable scale
                               lm_edge_anchors=lm_edge_anchors,
                               lm_grids=lm_grids,
                               lm_scale=2 / 256 * size)  # Normal random variable scale
        logger.debug("[AugConstants] warp constants: %s", retval)
        return retval

    @classmethod
    def from_config(cls,
                    processing_size: int,
                    batch_size: int) -> ConstantsAugmentation:
        """ Create a new dataclass instance from user config

        Parameters
        ----------
        processing_size
            The size of image to augment the data for
        batch_size
            The batch size that augmented data is being prepared for

        Returns
        -------
        The constructed augmentation constants
        """
        logger.debug("[AugConstants] Initializing %s(processing_size=%s, batch_size=%s)",
                     cls.__name__, processing_size, batch_size)
        retval = cls(color=cls._get_color(processing_size),
                     transform=cls._get_transform(processing_size),
                     warp=cls._get_warp(processing_size, batch_size))
        logger.debug(retval)
        return retval


class ImageAugmentation():
    """ Perform augmentation on batches of training images

    Applies random color and contrast adjustment, geometric transformation and warping to each
    batch before it is fed to a model. The behaviour for all stages comes from the constants held
    by :class:`ConstantsAugmentation`, which in turn read :file:`config.train.ini`

    Parameters
    ----------
    batch_size
        The number of images processed together through the augmentation functions at once
    processing_size
        The largest input or output size of the model, and therefore the size at which images are
        processed
    """
    def __init__(self, batch_size: int, processing_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._processing_size = processing_size
        self._batch_size = batch_size
        self._constants = ConstantsAugmentation.from_config(processing_size, batch_size)
        logger.debug("[Aug] Initialized %s", self.__class__.__name__)

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}(batch_size={self._batch_size}, "
                f"processing_size={self._processing_size})")

    # <<< COLOR AUGMENTATION >>> #
    def _random_lab(self, batch: np.ndarray) -> None:
        """ Perform random color/lightness adjustment in L*a*b* color space on a batch of images

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format of uint8 dtype
        """
        randoms = np.random.uniform(-self._constants.color.lab_adjust,
                                    self._constants.color.lab_adjust,
                                    size=(self._batch_size, 1, 1, 3)).astype("float32")
        logger.trace("[Aug] Random LAB adjustments: %s", randoms)  # type:ignore[attr-defined]
        # Iterating through the images and channels is much faster than numpy.where and slightly
        # faster than numexpr.where.
        for image, rand in zip(batch, randoms):
            for idx in range(rand.shape[-1]):
                adjustment = rand[:, :, idx]
                if adjustment >= 0:
                    image[:, :, idx] = ((255 - image[:, :, idx]) * adjustment) + image[:, :, idx]
                else:
                    image[:, :, idx] = image[:, :, idx] * (1 + adjustment)

    def _random_clahe(self, batch: np.ndarray) -> None:
        """ Randomly perform Contrast Limited Adaptive Histogram Equalization on a batch of images

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format of uint8 dtype
        """
        base_contrast = self._constants.color.clahe_base_contrast

        batch_random = np.random.rand(self._batch_size)
        indices = np.where(batch_random < self._constants.color.clahe_chance)[0]
        if not np.any(indices):
            return
        grid_bases = np.random.randint(self._constants.color.clahe_max_size + 1,
                                       size=indices.shape[0],
                                       dtype="uint8")
        grid_sizes = (grid_bases * (base_contrast // 2)) + base_contrast
        logger.trace("[Aug] Adjusting Contrast. Grid Sizes: %s",  # type:ignore[attr-defined]
                     grid_sizes)

        clahes = [cv2.createCLAHE(clipLimit=2.0,
                                  tileGridSize=(grid_size, grid_size))
                  for grid_size in grid_sizes]  # type:ignore[attr-defined]

        for idx, clahe in zip(indices, clahes):
            batch[idx, :, :, 0] = clahe.apply(batch[idx, :, :, 0], )

    def color_adjust(self, batch: np.ndarray) -> np.ndarray:
        """ Perform color augmentation on the passed in batch

        The color adjustment parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format of uint8 dtype

        Returns
        -------
        A 4-dimensional array of the same shape as :attr:`batch` with color augmentation applied
        """
        logger.trace("[Aug] Augmenting color")  # type:ignore[attr-defined]
        batch = batch_convert_color(batch, "BGR2LAB")
        self._random_lab(batch)
        self._random_clahe(batch)
        batch = batch_convert_color(batch, "LAB2BGR")
        return batch

    # <<< IMAGE AUGMENTATION >>> #
    def transform(self, batch: npt.NDArray[np.uint8], points: npt.NDArray[np.float32] | None
                  ) -> None:
        """ Perform random transformation on the passed in batch and any matching (x, y) points

        The transformation parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format
        points
            Any (x, y) points to transform, shaped as (batch_size, num_sides, 68, 2). Pass
            ``None`` when there are no points to transform
        """
        logger.trace("[Aug] Randomly transforming image")  # type:ignore[attr-defined]
        rotation = np.random.uniform(-self._constants.transform.rotation,
                                     self._constants.transform.rotation,
                                     size=self._batch_size).astype("float32")
        scale = np.random.uniform(1 - self._constants.transform.zoom,
                                  1 + self._constants.transform.zoom,
                                  size=self._batch_size).astype("float32")

        transform = np.random.uniform(-self._constants.transform.shift,
                                      self._constants.transform.shift,
                                      size=(self._batch_size, 2)).astype("float32")
        mats = batch_create_matrices(self._processing_size,
                                     rotation,
                                     scale=scale,
                                     translation=transform)

        for image, mat in zip(batch, mats[:, :2, :]):
            cv2.warpAffine(image,
                           mat,
                           (self._processing_size, self._processing_size),
                           dst=image,
                           borderMode=cv2.BORDER_REPLICATE)

        logger.trace("[Aug] Randomly transformed image")  # type:ignore[attr-defined]
        if points is None:
            return
        ones = np.ones((*points.shape[:-1], 1), dtype=points.dtype)
        pts_h = np.concatenate([points, ones], axis=-1)
        points[:] = np.einsum('nij,n...j->n...i', mats, pts_h)[..., :2]
        logger.trace("[Aug] Randomly transformed points")  # type:ignore[attr-defined]

    def random_flip(self, batch: npt.NDArray[np.uint8], points: npt.NDArray[np.float32] | None
                    ) -> None:
        """ Perform random horizontal flipping on the passed in batch.

        The probability of flipping an image is set in :file:`config.train.ini`

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format
        points
            Any (x, y) points to transform. Can be in any shape but the final dimension should be
            shape 2. ``None`` if there are no points to transform
        """
        logger.trace("[Aug] Randomly flipping image")  # type:ignore[attr-defined]
        randoms = np.random.rand(self._batch_size)
        indices = np.where(randoms <= self._constants.transform.flip)[0]
        batch[indices] = batch[indices, :, ::-1]
        logger.trace("[Aug] Randomly flipped %s images of %s",  # type:ignore[attr-defined]
                     len(indices), self._batch_size)
        if points is None:
            return
        points[indices, ..., 0] = (self._processing_size - 1) - points[indices, ..., 0]
        logger.trace("[Aug] Randomly flipped %s points: %s",  # type:ignore[attr-defined]
                     len(indices), format_array(points))

    def _random_warp(self, batch: np.ndarray) -> np.ndarray:
        """ Randomly warp the input batch

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format

        Returns
        -------
        A 4-dimensional array of the same shape as :attr:`batch` with warping applied
        """
        logger.trace("[Aug] Randomly warping batch")  # type:ignore[attr-defined]
        slices = self._constants.warp.slices
        rands = np.random.normal(size=(self._batch_size, 2, 5, 5),
                                 scale=self._constants.warp.scale).astype("float32")
        batch_maps = ne.evaluate("m + r", local_dict={"m": self._constants.warp.maps, "r": rands})

        interpolators = np.array([[cv2.resize(map_, self._constants.warp.pad)[slices, slices]
                                  for map_ in maps]
                                 for maps in batch_maps])
        warped_batch = np.array([cv2.remap(image,
                                           interpolator[0],
                                           interpolator[1],
                                           cv2.INTER_LINEAR)
                                 for image, interpolator in zip(batch, interpolators)])

        logger.trace("[Aug] Warped image shape: %s",  # type:ignore[attr-defined]
                     warped_batch.shape)
        return warped_batch

    def _random_warp_landmarks(self,
                               batch: np.ndarray,
                               batch_src_points: np.ndarray,
                               batch_dst_points: np.ndarray) -> np.ndarray:
        """ Warp the image to a similar set of landmarks from the opposite side

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format
        batch_src_points
            A batch of 68 point landmarks for the source faces, shaped as (`batchsize`, `68`, `2`)
        batch_dst_points
            A batch of randomly chosen closest matching destination face landmarks, shaped as
            (`batchsize`, `68`, `2`)

        Returns
        -------
        A 4-dimensional array of the same shape as :attr:`batch` with warping applied
        """
        logger.trace("[Aug] Randomly warping landmarks")  # type:ignore[attr-defined]
        edge_anchors = self._constants.warp.lm_edge_anchors
        grids = self._constants.warp.lm_grids

        batch_dst = batch_dst_points + np.random.normal(size=batch_dst_points.shape,
                                                        scale=self._constants.warp.lm_scale)

        face_cores = [cv2.convexHull(np.concatenate([src[17:], dst[17:]], axis=0))
                      for src, dst in zip(batch_src_points.astype("int32"),
                                          batch_dst.astype("int32"))]

        batch_src = np.append(batch_src_points, edge_anchors, axis=1)
        batch_dst = np.append(batch_dst, edge_anchors, axis=1)

        rem_indices = [list(set(idx for fpl in (src, dst)
                                for idx, (pty, ptx) in enumerate(fpl)
                                if cv2.pointPolygonTest(face_core, (pty, ptx), False) >= 0))
                       for src, dst, face_core in zip(batch_src[:, :18, :],
                                                      batch_dst[:, :18, :],
                                                      face_cores)]
        lm_batch_src = [np.delete(src, indices, axis=0)
                        for indices, src in zip(rem_indices, batch_src)]
        lm_batch_dst = [np.delete(dst, indices, axis=0)
                        for indices, dst in zip(rem_indices, batch_dst)]

        grid_z = np.array([griddata(dst, src, (grids[0], grids[1]), method="linear")
                           for src, dst in zip(lm_batch_src, lm_batch_dst)])
        maps = grid_z.reshape((self._batch_size,
                               self._processing_size,
                               self._processing_size,
                               2)).astype("float32")

        warped_batch = np.array([cv2.remap(image,
                                           map_[..., 1],
                                           map_[..., 0],
                                           cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_TRANSPARENT)
                                 for image, map_ in zip(batch, maps)])
        logger.trace("[Aug] Warped batch shape: %s",  # type:ignore[attr-defined]
                     warped_batch.shape)
        return warped_batch

    def warp(self,
             batch: np.ndarray,
             to_landmarks: bool = False,
             batch_src_points: np.ndarray | None = None,
             batch_dst_points: np.ndarray | None = None) -> np.ndarray:
        """ Perform random warping on the passed in batch by one of two methods

        The color adjustment parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format
        to_landmarks
            If ``False`` perform standard random warping of the input image. If ``True`` perform
            warping to semi-random corresponding landmarks from the other side. Default: ``False``
        batch_src_points, optional
            Only used when :attr:`to_landmarks` is ``True``. A batch of 68 point landmarks for the
            source faces, shaped as (`batchsize`, `68`, `2`). Default: ``None``
        batch_dst_points, optional
            Only used when :attr:`to_landmarks` is ``True``. A batch of randomly chosen closest
            matching destination face landmarks, shaped as (`batchsize`, `68`, `2`).
            Default: ``None``

        Returns
        -------
        A 4-dimensional array of the same shape as :attr:`batch` with warping applied
        """
        if to_landmarks:
            assert batch_src_points is not None
            assert batch_dst_points is not None
            return self._random_warp_landmarks(batch, batch_src_points, batch_dst_points)
        return self._random_warp(batch)


__all__ = get_module_objects(__name__)

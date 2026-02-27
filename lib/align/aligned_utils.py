#!/usr/bin/env python3
""" Tools for working with aligned faces and aligned masks """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.utils import get_module_objects

from .constants import EXTRACT_RATIOS

logger = logging.getLogger(__name__)


if T.TYPE_CHECKING:
    from .constants import CenteringType


def get_adjusted_center(image_size: int,
                        source_offset: np.ndarray,
                        target_offset: np.ndarray,
                        source_centering: CenteringType,
                        y_offset: float) -> np.ndarray:
    """ Obtain the correct center of a face extracted image to translate between two different
    extract centerings.

    Parameters
    ----------
    image_size: int
        The size of the image at the given :attr:`source_centering`
    source_offset: :class:`numpy.ndarray`
        The pose offset to translate a base extracted face to source centering
    target_offset: :class:`numpy.ndarray`
        The pose offset to translate a base extracted face to target centering
    source_centering: ["face", "head", "legacy"]
        The centering of the source image
    y_offset: float
        Amount to additionally offset the center of the image along the y-axis

    Returns
    -------
    :class:`numpy.ndarray`
        The center point of the image at the given size for the target centering
    """
    source_size = image_size - (image_size * EXTRACT_RATIOS[source_centering])
    offset = target_offset - source_offset - [0., y_offset]
    offset *= source_size
    center = np.rint(offset + image_size / 2).astype("int32")
    logger.trace(  # type:ignore[attr-defined]
        "image_size: %s, source_offset: %s, target_offset: %s, source_centering: '%s', "
        "y_offset: %s, adjusted_offset: %s, center: %s",
        image_size, source_offset, target_offset, source_centering, y_offset, offset, center)
    return center


def get_centered_size(source_centering: CenteringType,
                      target_centering: CenteringType,
                      size: int,
                      coverage_ratio: float = 1.0) -> int:
    """ Obtain the size of a cropped face from an aligned image.

    Given an image of a certain dimensions, returns the dimensions of the sub-crop within that
    image for the requested centering at the requested coverage ratio

    Notes
    -----
    `"legacy"` places the nose in the center of the image (the original method for aligning).
    `"face"` aligns for the nose to be in the center of the face (top to bottom) but the center
    of the skull for left to right. `"head"` places the center in the middle of the skull in 3D
    space.

    The ROI in relation to the source image is calculated by rounding the padding of one side
    to the nearest integer then applying this padding to the center of the crop, to ensure that
    any dimensions always have an even number of pixels.

    Parameters
    ----------
    source_centering: ["head", "face", "legacy"]
        The centering that the original image is aligned at
    target_centering: ["head", "face", "legacy"]
        The centering that the sub-crop size should be obtained for
    size: int
        The size of the source image to obtain the cropped size for
    coverage_ratio: float, optional
        The coverage ratio to be applied to the target image. Default: `1.0`

    Returns
    -------
    int
        The pixel size of a sub-crop image from a full head aligned image with the given coverage
        ratio
    """
    if source_centering == target_centering and coverage_ratio == 1.0:
        src_size: float | int = size
        retval = size
    else:
        src_size = size - (size * EXTRACT_RATIOS[source_centering])
        retval = 2 * int(np.rint((src_size / (1 - EXTRACT_RATIOS[target_centering])
                                 * coverage_ratio) / 2))
    logger.trace(  # type:ignore[attr-defined]
        "source_centering: %s, target_centering: %s, size: %s, coverage_ratio: %s, "
        "source_size: %s, crop_size: %s",
        source_centering, target_centering, size, coverage_ratio, src_size, retval)
    return retval


def get_matrix_scaling(matrix: np.ndarray) -> tuple[int, int]:
    """ Given a matrix, return the cv2 Interpolation method and inverse interpolation method for
    applying the matrix on an image.

    Parameters
    ----------
    matrix: :class:`numpy.ndarray`
        The transform matrix to return the interpolator for

    Returns
    -------
    tuple
        The interpolator and inverse interpolator for the given matrix. This will be (Cubic, Area)
        for an upscale matrix and (Area, Cubic) for a downscale matrix
    """
    x_scale = np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[0, 1] * matrix[0, 1])
    if x_scale == 0:
        y_scale = 0.
    else:
        y_scale = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) / x_scale
    avg_scale = (x_scale + y_scale) * 0.5
    if avg_scale >= 1.:
        interpolators = cv2.INTER_CUBIC, cv2.INTER_AREA
    else:
        interpolators = cv2.INTER_AREA, cv2.INTER_CUBIC
    logger.trace("interpolator: %s, inverse interpolator: %s",  # type:ignore[attr-defined]
                 interpolators[0], interpolators[1])
    return interpolators


def transform_image(image: np.ndarray,
                    matrix: np.ndarray,
                    size: int,
                    padding: int = 0) -> np.ndarray:
    """ Perform transformation on an image, applying the given size and padding to the matrix.

    Parameters
    ----------
    image: :class:`numpy.ndarray`
        The image to transform
    matrix: :class:`numpy.ndarray`
        The transformation matrix to apply to the image
    size: int
        The final size of the transformed image
    padding: int, optional
        The amount of padding to apply to the final image. Default: `0`

    Returns
    -------
    :class:`numpy.ndarray`
        The transformed image
    """
    logger.trace("image shape: %s, matrix: %s, size: %s. padding: %s",  # type:ignore[attr-defined]
                 image.shape, matrix, size, padding)
    # transform the matrix for size and padding
    mat = matrix * (size - 2 * padding)
    mat[:, 2] += padding

    # transform image
    interpolators = get_matrix_scaling(mat)
    retval = cv2.warpAffine(image, mat, (size, size), flags=interpolators[0])
    logger.trace("transformed matrix: %s, final image shape: %s",  # type:ignore[attr-defined]
                 mat, image.shape)
    return retval


__all__ = get_module_objects(__name__)

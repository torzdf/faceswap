#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.training.data.augmentation` """
from __future__ import annotations

import typing as T

import numpy as np
import pytest

from lib.training.data.augmentation import ConstantsAugmentation, ImageAugmentation
from plugins.train.trainer import trainer_config as cfg

from tests.lib.config.helpers import patch_config  # pylint:disable=unused-import  # noqa[F401]

# pylint:disable=protected-access,redefined-outer-name,duplicate-code


# =============================================================================
# Constants
# =============================================================================

MODULE_PREFIX = "lib.training.data.augmentation"

DEFAULT_CONFIG = {"color_clahe_chance": 0,
                  "color_clahe_max_size": 4,
                  "color_lightness": 0,
                  "color_ab": 0,
                  "rotation_range": 0,
                  "zoom_amount": 0,
                  "shift_range": 0,
                  "flip_chance": 0}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def make_augment(patch_config: T.Callable[[object, dict], None],  # noqa: F811
                 ) -> T.Callable[..., ImageAugmentation]:
    """ Patch config and construct an augmentation instance with the given overrides """
    def _build(config: dict | None = None,
               batch_size: int = 8,
               processing_size: int = 64) -> ImageAugmentation:
        """ Merge overrides onto the defaults and build the instance """
        merged = {**DEFAULT_CONFIG, **(config or {})}
        patch_config(cfg.Augmentation, merged)
        return ImageAugmentation(batch_size=batch_size, processing_size=processing_size)

    return _build


@pytest.fixture()
def make_batch() -> T.Callable[[int, int], np.ndarray]:
    """ Build a deterministic uint8 BGR image batch of the requested shape """
    def _build(batch_size: int, size: int) -> np.ndarray:
        """ Return random in-range pixels under a fixed seed for reproducibility """
        rng = np.random.default_rng(0)

        return (rng.random((batch_size, size, size, 3)) * 255).astype("uint8")

    return _build


# =============================================================================
# ConstantsAugmentation - _get_clahe
# =============================================================================

_CLAHE_CASES = (({"color_clahe_chance": 10, "color_clahe_max_size": 4}, 64),
                ({"color_clahe_chance": 50, "color_clahe_max_size": 8}, 256))


@pytest.mark.parametrize(("config", "size"), _CLAHE_CASES)
def test_clahe_values_from_config(config: dict[str, T.Any],
                                  size: int,
                                  make_augment: T.Callable[..., ImageAugmentation]) -> None:
    """ Derive clahe constants from the patched user config """
    make_augment(config)

    base = max(2, size // 128)
    clahe_base_contrast, clahe_chance, clahe_max_size = ConstantsAugmentation._get_clahe(size)

    assert clahe_base_contrast == base
    assert clahe_chance == config["color_clahe_chance"] / 100.
    assert clahe_max_size == config["color_clahe_max_size"]


# =============================================================================
# ConstantsAugmentation - _get_lab
# =============================================================================

def test_lab_values_track_lightness_ab(make_augment: T.Callable[..., ImageAugmentation]) -> None:
    """ Map lightness and ab amounts onto the three L*A*B adjustment channels """
    make_augment({"color_lightness": 40, "color_ab": 12})

    lab_adjust = ConstantsAugmentation._get_lab()

    assert isinstance(lab_adjust, np.ndarray)
    assert lab_adjust.dtype == np.float32
    assert lab_adjust.shape == (3,)
    assert lab_adjust[0] == 40 / 100.
    assert lab_adjust[1] == 12 / 100.
    assert lab_adjust[2] == 12 / 100.


# =============================================================================
# ConstantsAugmentation - _get_color
# =============================================================================

def test_color_aggregates_clahe_and_lab(make_augment: T.Callable[..., ImageAugmentation]) -> None:
    """ Combine clahe and lab constants into a single colour constant set """
    make_augment({"color_clahe_chance": 50, "color_clahe_max_size": 8,
                  "color_lightness": 30, "color_ab": 12})

    color = ConstantsAugmentation._get_color(256)

    assert color.clahe_base_contrast == max(2, 256 // 128)
    assert color.clahe_chance == 50 / 100.
    assert color.clahe_max_size == 8
    assert color.lab_adjust[0] == 30 / 100.
    assert color.lab_adjust[1] == 12 / 100.


# =============================================================================
# ConstantsAugmentation - _get_transform
# =============================================================================

def test_transform_constants_scaled(make_augment: T.Callable[..., ImageAugmentation]) -> None:
    """ Derive transform constants, scaling shift and amounts by config values """
    make_augment({"rotation_range": 45, "zoom_amount": 250,
                  "shift_range": 100, "flip_chance": 30})

    transform = ConstantsAugmentation._get_transform(64)

    assert transform.rotation == 45
    assert transform.zoom == 250 / 100.
    assert transform.shift == (100 / 100.) * 64
    assert transform.flip == 30 / 100.


# =============================================================================
# ConstantsAugmentation - _get_warp_to_landmarks
# =============================================================================

def test_warp_landmark_shapes_by_batch() -> None:
    """ Shape the edge anchors and grids according to batch size and image size """
    size = 64
    batch_size = 8
    edge_anchors, grids = ConstantsAugmentation._get_warp_to_landmarks(size, batch_size)

    assert edge_anchors.shape == (batch_size, 8, 2)
    assert edge_anchors.dtype == np.int32
    assert grids.shape == (2, size, size)
    assert grids.dtype == np.float32

    phf = (size // 2) - 1
    expected_row = np.array([[0, 0], [0, size - 1], [size - 1, size - 1], [size - 1, 0],
                             [phf, 0], [phf, size - 1], [size - 1, phf], [0, phf]], dtype="int32")
    assert np.array_equal(edge_anchors[0], expected_row)


# =============================================================================
# ConstantsAugmentation - _get_warp
# =============================================================================

def test_warp_maps_stacked_xy() -> None:
    """ Stack the x and y warp maps and derive pad, slice and scale values """
    size = 64
    batch_size = 8
    warp = ConstantsAugmentation._get_warp(size, batch_size)

    assert warp.maps.shape == (batch_size, 2, 5, 5)
    assert warp.pad == (int(1.25 * size), int(1.25 * size))
    assert warp.slices.start == int(1.25 * size) // 10
    assert warp.slices.stop == -(int(1.25 * size) // 10)
    assert warp.scale == 5 / 256 * size
    assert warp.lm_scale == 2 / 256 * size


# =============================================================================
# ConstantsAugmentation - from_config
# =============================================================================

def test_from_config_constructs_constants(make_augment: T.Callable[..., ImageAugmentation]
                                          ) -> None:
    """ Aggregate colour, transform and warp constants from the user config """
    make_augment({"rotation_range": 45})

    constants = ConstantsAugmentation.from_config(128, 8)

    assert isinstance(constants.color.clahe_chance, float)
    assert constants.transform.rotation == 45
    assert constants.warp.scale == 5 / 256 * 128


# =============================================================================
# ConstantsWarp - __repr__
# =============================================================================

def test_repr_hides_warp_arrays() -> None:
    """ Keep large arrays out of the string representation """
    text = repr(ConstantsAugmentation._get_warp(64, 8))

    assert text.startswith("ConstantsWarp(")
    assert len(text) < 500


# =============================================================================
# ImageAugmentation - __init__ / __repr__
# =============================================================================

def test_init_stores_sizes(make_augment: T.Callable[..., ImageAugmentation]) -> None:
    """ Store the batch and processing sizes while building the constants """
    instance = make_augment({"rotation_range": 45}, batch_size=8, processing_size=64)

    assert instance._batch_size == 8
    assert instance._processing_size == 64


def test_reflects_sizes_in_repr() -> None:
    """ Render the configured sizes in the string representation """
    instance = ImageAugmentation(batch_size=16, processing_size=128)

    text = repr(instance)

    assert "batch_size=16" in text
    assert "processing_size=128" in text


# =============================================================================
# ImageAugmentation - _random_lab / _random_clahe / color_adjust
# =============================================================================

def test_lab_zeroed_is_identity(make_augment: T.Callable[..., ImageAugmentation],
                                make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Leave the batch untouched when no lightness or ab adjustment is configured """
    instance = make_augment({"color_lightness": 0, "color_ab": 0})
    batch = make_batch(8, 64)
    reference = batch.copy()

    instance._random_lab(batch)

    assert np.array_equal(batch, reference)


def test_lab_nonzero_changes_pixels(make_augment: T.Callable[..., ImageAugmentation],
                                    make_batch) -> None:
    """ Alter pixel values when lightness or ab adjustment is configured """
    instance = make_augment({"color_lightness": 50, "color_ab": 25})
    batch = make_batch(8, 64)
    reference = batch.copy()

    instance._random_lab(batch)

    assert not np.array_equal(batch, reference)


def test_clahe_off_is_identity(make_augment: T.Callable[..., ImageAugmentation],
                               make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Leave the batch untouched when clahe is never applied """
    instance = make_augment({"color_clahe_chance": 0})
    batch = make_batch(8, 64)
    reference = batch.copy()

    instance._random_clahe(batch)

    assert np.array_equal(batch, reference)


def test_clahe_on_changes_l_channel(make_augment: T.Callable[..., ImageAugmentation],
                                    make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Alter the first channel when clahe is applied to every image """
    instance = make_augment({"color_clahe_chance": 100})
    batch = make_batch(8, 64)
    reference = batch.copy()

    instance._random_clahe(batch)

    assert not np.array_equal(batch[:, :, :, 0], reference[:, :, :, 0])


def test_color_adjust_keeps_shape_type(make_augment: T.Callable[..., ImageAugmentation],
                                       make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Preserve the batch shape and dtype while adjusting colour """
    instance = make_augment({"color_lightness": 50, "color_ab": 25, "color_clahe_chance": 100})
    batch = make_batch(8, 64)

    output = instance.color_adjust(batch.copy())

    assert output.shape == batch.shape
    assert output.dtype == np.uint8
    assert output.min() >= 0
    assert output.max() <= 255


def test_color_adjust_alters_data(make_augment: T.Callable[..., ImageAugmentation],
                                  make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Change the pixel data when colour augmentation is configured """
    instance = make_augment({"color_lightness": 50, "color_ab": 25, "color_clahe_chance": 100})
    batch = make_batch(8, 64)

    output = instance.color_adjust(batch.copy())

    assert not np.array_equal(output, batch)


# =============================================================================
# ImageAugmentation - transform
# =============================================================================

def test_transform_zeroed_is_identity(make_augment: T.Callable[..., ImageAugmentation],
                                      make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Leave images and points unchanged when no transform is configured """
    instance = make_augment()
    batch = make_batch(8, 64)
    reference = batch.copy()
    points = np.random.default_rng(0).random((8, 2, 2)).astype("float32")
    point_reference = points.copy()

    result = instance.transform(batch, points)

    assert result is None
    assert np.array_equal(batch, reference)
    assert np.array_equal(points, point_reference)


def test_transform_moves_images_pts(make_augment: T.Callable[..., ImageAugmentation],
                                    make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Warp images and shift points when a transform is configured """
    instance = make_augment({"rotation_range": 30, "zoom_amount": 20, "shift_range": 10})
    batch = make_batch(8, 64)
    reference = batch.copy()
    points = np.random.default_rng(0).random((8, 2, 2)).astype("float32")
    point_reference = points.copy()

    result = instance.transform(batch, points)

    assert result is None
    assert not np.array_equal(batch, reference)
    assert not np.array_equal(points, point_reference)


# =============================================================================
# ImageAugmentation - random_flip
# =============================================================================

def test_flip_all_reverses_x_coord(make_augment: T.Callable[..., ImageAugmentation],
                                   make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Reverse columns and x-coordinates when every image is flipped """
    instance = make_augment({"flip_chance": 100})
    batch = make_batch(8, 64)
    reference = batch.copy()
    size = instance._processing_size
    points = np.random.default_rng(0).random((8, 2, 2)).astype("float32") * (size - 1)
    point_reference = points.copy()

    result = instance.random_flip(batch, points)

    assert result is None
    assert np.array_equal(batch, reference[:, :, ::-1])
    assert np.array_equal(points[..., 0], (size - 1) - point_reference[..., 0])


def test_flip_none_keeps_images_pts(make_augment: T.Callable[..., ImageAugmentation],
                                    make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Leave images and points unchanged when nothing is flipped """
    instance = make_augment({"flip_chance": 0})
    batch = make_batch(8, 64)
    reference = batch.copy()
    size = instance._processing_size
    points = np.random.default_rng(0).random((8, 2, 2)).astype("float32") * (size - 1)
    point_reference = points.copy()

    result = instance.random_flip(batch, points)

    assert result is None
    assert np.array_equal(batch, reference)
    assert np.array_equal(points, point_reference)


# =============================================================================
# ImageAugmentation - _random_warp / _random_warp_landmarks / warp
# =============================================================================

def test_warp_preserves_shape_distorts(make_augment: T.Callable[..., ImageAugmentation],
                                       make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Return a same-shaped uint8 batch with the image warped """
    instance = make_augment()
    batch = make_batch(8, 64)
    reference = batch.copy()

    output = instance._random_warp(batch)

    assert output.shape == batch.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, reference)


def test_landmark_warp_distorts_batch(make_augment: T.Callable[..., ImageAugmentation],
                                      make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Warp the same-shaped uint8 batch using both landmark point sets """
    instance = make_augment({}, batch_size=2)
    batch = make_batch(2, 64)
    src_points = np.random.default_rng(0).random((2, 68, 2)).astype("float32") * 64
    dst_points = np.random.default_rng(1).random((2, 68, 2)).astype("float32") * 64

    output = instance._random_warp_landmarks(batch, src_points, dst_points)

    assert output.shape == batch.shape
    assert output.dtype == np.uint8


def test_warp_routes_standard_flag(make_augment: T.Callable[..., ImageAugmentation],
                                   make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Warp images through the standard path when landmark warping is not requested """
    instance = make_augment({}, batch_size=2)
    batch = make_batch(2, 64)

    output = instance.warp(batch, to_landmarks=False)

    assert output.shape == batch.shape


def test_warp_routes_landmark_flag(make_augment: T.Callable[..., ImageAugmentation],
                                   make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Warp end-to-end through the landmark path when requested """
    instance = make_augment({}, batch_size=2)
    batch = make_batch(2, 64)
    src_points = np.random.default_rng(0).random((2, 68, 2)).astype("float32") * 64
    dst_points = np.random.default_rng(1).random((2, 68, 2)).astype("float32") * 64

    output = instance.warp(batch, to_landmarks=True,
                           batch_src_points=src_points, batch_dst_points=dst_points)

    assert output.shape == batch.shape


def test_landmark_warp_requires_points(make_augment: T.Callable[..., ImageAugmentation],
                                       make_batch: T.Callable[[int, int], np.ndarray]) -> None:
    """ Fail when landmark warping is requested without the matching points """
    instance = make_augment()
    batch = make_batch(2, 64)

    with pytest.raises(AssertionError):
        instance.warp(batch, to_landmarks=True)

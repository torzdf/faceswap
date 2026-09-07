#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.training.data.data_set` """
from __future__ import annotations

import shutil
import typing as T
from pathlib import Path

import numpy as np
import pytest
import torch

from lib.training.data.data_set import (MultiDataset, PreviewSet, TrainSet, _MaskProcessing,
                                        get_label, get_sorted_images, to_float32)
from lib.utils import FaceswapError
from plugins.train import train_config as cfg
from tests.lib.config.helpers import patch_config  # pylint:disable=unused-import  # noqa[F401]

# pylint:disable=protected-access,redefined-outer-name,import-outside-toplevel


# =============================================================================
# Constants
# =============================================================================

MODULE_PREFIX = "lib.training.data.data_set"

_SOURCE_FACES = Path(__file__).resolve().parents[3] / "data" / "imgs" / "faces"

_DEFAULT_LOSS_CONFIG = {"mask_dilation": 2,
                        "mask_blur_kernel": 3,
                        "mask_threshold": 10,
                        "mask_type": None,
                        "learn_mask": False,
                        "penalized_mask_loss": False,
                        "eye_multiplier": 1,
                        "mouth_multiplier": 1}

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def patch_train_cfg(patch_config: T.Callable[[object, dict], None],  # noqa: F811
                    ) -> T.Callable[[dict[str, T.Any] | None], None]:
    """ Patch the training config with known values before dataset construction """
    def _apply(loss_overrides: dict[str, T.Any] | None = None) -> None:
        """ Overwrite centering/coverage/y_offset and the full Loss namespace """
        patch_config(cfg, {"centering": "face", "coverage": 100.0, "vertical_offset": 0})
        loss = {**_DEFAULT_LOSS_CONFIG}
        if loss_overrides:
            loss.update(loss_overrides)
        patch_config(cfg.Loss, loss)

    return _apply


@pytest.fixture()
def train_set(patch_train_cfg) -> T.Callable[..., TrainSet]:  # noqa: F811
    """ Build a TrainSet over the given folder with patched config and size overrides """
    def _build(image_folder: Path,
               size: int = 128,
               loss_overrides: dict[str, T.Any] | None = None) -> TrainSet:
        """ Patch config then construct and return a TrainSet instance """
        patch_train_cfg(loss_overrides)
        return TrainSet("A", image_folder, size=size)

    return _build


@pytest.fixture()
def preview_set(patch_train_cfg) -> T.Callable[..., PreviewSet]:  # noqa: F811
    """ Build a PreviewSet over the given folder with patched config and color overrides """
    def _build(image_folder: Path,
               input_size: int = 64,
               output_size: int = 128,
               color_order: T.Literal["bgr", "rgb"] = "bgr",
               loss_overrides: dict[str, T.Any] | None = None) -> PreviewSet:
        """ Patch config then construct and return a PreviewSet instance """
        patch_train_cfg(loss_overrides)
        return PreviewSet("B", image_folder, input_size=input_size,
                          output_size=output_size, color_order=color_order)

    return _build


@pytest.fixture()
def make_mask_proc(
    patch_config: T.Callable[[object, dict], None],  # noqa: F811
) -> T.Callable[..., _MaskProcessing]:
    """ Build a _MaskProcessing instance with known config values for direct tests """
    def _build(side: str = "A", size: int = 128) -> _MaskProcessing:
        """ Patch the mask-loss config and return a configured _MaskProcessing object """
        patch_config(cfg, {"centering": "head", "coverage": 87.5, "vertical_offset": 10})
        patch_config(cfg.Loss, {"mask_dilation": 2, "mask_blur_kernel": 3, "mask_threshold": 10})
        return _MaskProcessing(side, size, cfg.coverage() / 100., cfg.centering(),
                               cfg.vertical_offset() / 100.)

    return _build


@pytest.fixture()
def faces(tmp_path: Path) -> Path:
    """ Populate a temp folder with real extracted-face PNGs for dataset loading tests """
    for name, real in (("a.png", "test_img1_0.png"), ("b.png", "test_img3_0.png")):
        shutil.copy(_SOURCE_FACES / real, tmp_path / name)
    return tmp_path


# =============================================================================
# Tier 1: Pure / Low-level Functions
# =============================================================================

# =============================================================================
# to_float32
# =============================================================================


def test_casts_uint8_to_unit_range() -> None:
    """ Cast a uint8 0-255 array into a float32 0.0-1.0 array preserving shape and dtype """
    values = np.array([0, 1, 127, 128, 254, 255], dtype=np.uint8)

    result = to_float32(values)

    assert result.dtype == np.float32
    assert result.shape == values.shape
    assert np.allclose(result, [0.0, 1 / 255, 127 / 255, 128 / 255, 254 / 255, 1.0])


def test_casts_multidimensional_batch() -> None:
    """ Cast a multi-dimensional uint8 batch to float32 in the unit range without layout change """
    batch = (np.random.default_rng(0).random((2, 4, 4, 3)) * 255).astype(np.uint8)

    result = to_float32(batch)

    assert result.dtype == np.float32
    assert result.shape == batch.shape
    assert result.min() >= 0.0 and result.max() <= 1.0


# =============================================================================
# get_label
# =============================================================================


def test_cycles_through_alphabet_in_order() -> None:
    """ Return the identity alphabet A-Z in order for a request within the first 26 identities """
    expected = "".join(chr(i) for i in range(65, 91))

    labels = [get_label(index, 26) for index in range(26)]

    assert "".join(labels) == expected


def test_next_identity_wraps_around() -> None:
    """ Advance to the next identity and wrap back to A when already at the final one """
    assert get_label(0, 26, next_identity=True) == "B"
    assert get_label(25, 26, next_identity=True) == "A"


def test_extends_alphabet_with_digits() -> None:
    """ Append digits then letters once the identity count exceeds the 26-letter alphabet """
    extended = "".join(get_label(index, 37) for index in range(40))

    assert extended[26:36] == "0123456789"
    assert extended[36] == "a"


def test_raises_when_too_many_identities() -> None:
    """ Fail when more identities are requested than the alphabet can represent """
    with pytest.raises(FaceswapError):
        get_label(0, 63)


# =============================================================================
# get_sorted_images
# =============================================================================


def test_returns_only_png_paths_sorted(tmp_path: Path) -> None:
    """ Return full paths to only the PNG files within a folder sorted alphabetically """
    for name in ["c.png", "a.png", "b.txt", "d.png", "e.PNG.bak", "b.png"]:
        (tmp_path / name).write_text("", encoding="utf-8")

    paths = [Path(p).name for p in get_sorted_images(str(tmp_path))]

    assert paths == ["a.png", "b.png", "c.png", "d.png"]


# =============================================================================
# _MaskProcessing - construction, repr and mask-existence checks
# =============================================================================


def test_repr_lists_constructor_parameters(
        make_mask_proc: T.Callable[..., _MaskProcessing]) -> None:
    """ Render the side, size, coverage, centering and y-offset used to build the processor """
    assert repr(make_mask_proc("A", 128)) == (
        "_MaskProcessing(side='A', size=128, coverage_ratio=0.875, "
        "centering='head', y_offset=0.1)")


def test_missing_header_mask_raises(make_mask_proc: T.Callable[..., _MaskProcessing]) -> None:
    """ Raise a FaceswapError listing the available masks when only header masks are requested """
    processor = make_mask_proc()

    with pytest.raises(FaceswapError) as excinfo:
        processor._check_mask_exists([], "face", "missing.png")

    assert "missing.png" in str(excinfo.value)
    assert "components" in str(excinfo.value)


def test_landmark_mask_types_are_considered_present(
        make_mask_proc: T.Callable[..., _MaskProcessing]) -> None:
    """ Accept landmark-derived mask types even when no header masks are stored for the face """
    processor = make_mask_proc()

    processor._check_mask_exists([], "eye", "face.png")
    processor._check_mask_exists([], "mouth", "face.png")


# =============================================================================
# TrainSet - configuration and configured-mask selection
# =============================================================================


def test_constructor_applies_config_and_shape(train_set: T.Callable[..., TrainSet],
                                              faces: Path) -> None:
    """ Build a TrainSet applying config into side, mask types and output shape """
    dataset = train_set(faces, size=128, loss_overrides={"mask_type": "components",
                                                         "learn_mask": True})

    assert dataset._side == "A"
    assert dataset._mask_types == ["components"]
    assert dataset._out_shape == (128, 128, 4)
    assert dataset._size == 128


def test_no_masks_when_loss_disabled(train_set: T.Callable[..., TrainSet], faces: Path) -> None:
    """ Configure no masks when loss based masking and learn-mask are both disabled """
    dataset = train_set(faces, size=128)

    assert dataset._mask_types == []
    assert dataset._out_shape == (128, 128, 3)


def test_eyes_and_mouth_added_with_penalty(
        train_set: T.Callable[..., TrainSet], faces: Path) -> None:
    """ Add eye and mouth masks when penalized mask loss is enabled with multipliers above one """
    dataset = train_set(faces, size=128, loss_overrides={"mask_type": "face",
                                                         "penalized_mask_loss": True,
                                                         "eye_multiplier": 2,
                                                         "mouth_multiplier": 2})

    assert dataset._mask_types == ["face", "eye", "mouth"]


def test_only_face_type_without_penalty(train_set: T.Callable[..., TrainSet], faces: Path) -> None:
    """ Keep just the face mask type when learn-mask is used without penalized mask loss """
    dataset = train_set(faces, size=128, loss_overrides={"mask_type": "components",
                                                         "learn_mask": True})

    assert dataset._mask_types == ["components"]


# =============================================================================
# TrainSet - __getitem__
# =============================================================================


def test_returns_image_and_index_without_masks(
        train_set: T.Callable[..., TrainSet], faces: Path) -> None:
    """ Return the stacked training image and its file index without any mask channels """
    dataset = train_set(faces, size=128)

    image, index = dataset[0]

    assert index == 0
    assert isinstance(image, np.ndarray)
    assert image.shape == (128, 128, 3)
    assert image.dtype == np.uint8
    assert bool(image[..., :3].sum()) > 0


def test_stacks_mask_into_output(train_set: T.Callable[..., TrainSet], faces: Path) -> None:
    """ Stack one configured mask into the output channel when loss based masking is enabled """
    dataset = train_set(faces, size=128, loss_overrides={"mask_type": "components",
                                                         "learn_mask": True})

    image, index = dataset[0]

    assert index == 0
    assert image.shape == (128, 128, 4)
    assert image.dtype == np.uint8
    assert image[..., 3].max() > 0


# =============================================================================
# PreviewSet - configuration and configured-mask selection
# =============================================================================


def test_no_masks_by_default(preview_set: T.Callable[..., PreviewSet], faces: Path) -> None:
    """ Configure no masks for a preview set when loss based masking is disabled """
    dataset = preview_set(faces, input_size=64, output_size=128)

    assert dataset._mask_types == []


def test_single_face_mask_type(preview_set: T.Callable[..., PreviewSet], faces: Path) -> None:
    """ Configure only the face mask type for a preview set when learn-mask is enabled """
    dataset = preview_set(faces, input_size=64, output_size=128,
                          loss_overrides={"mask_type": "face", "learn_mask": True})

    assert dataset._mask_types == ["face"]


def test_full_size_rounded_to_even(preview_set: T.Callable[..., PreviewSet], faces: Path) -> None:
    """ Round the full preview size to the nearest even value based on coverage and output size """
    dataset = preview_set(faces, input_size=64, output_size=128)

    assert dataset._full_size == 128


# =============================================================================
# PreviewSet - __getitem__
# =============================================================================


def test_returns_feed_and_target_tensors(
        preview_set: T.Callable[..., PreviewSet], faces: Path) -> None:
    """ Return float32 feed and target tensors for the model input and full coverage """
    dataset = preview_set(faces, input_size=64, output_size=128,
                          loss_overrides={"mask_type": "components", "learn_mask": True})

    feed, target = dataset[0]

    assert isinstance(feed, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert feed.shape == (3, 64, 64)
    assert feed.dtype == torch.float32
    assert target.shape == (128, 128, 4)
    assert target.dtype == torch.float32


def test_swaps_channels_for_rgb(preview_set: T.Callable[..., PreviewSet], faces: Path) -> None:
    """ Reverse the colour channels of both feed and target when rgb color order is requested """
    bgr = preview_set(faces, input_size=64, output_size=128, color_order="bgr",
                      loss_overrides={"mask_type": "components", "learn_mask": True})
    rgb = preview_set(faces, input_size=64, output_size=128, color_order="rgb",
                      loss_overrides={"mask_type": "components", "learn_mask": True})

    bgr_feed, _ = bgr[0]
    rgb_feed, _ = rgb[0]

    assert np.array_equal(rgb_feed[0].numpy(), bgr_feed[2].numpy())
    assert np.array_equal(rgb_feed[2].numpy(), bgr_feed[0].numpy())


def test_alpha_channel_holds_mask(preview_set: T.Callable[..., PreviewSet], faces: Path) -> None:
    """ Populate the target alpha channel with the computed mask rather than a constant value """
    dataset = preview_set(faces, input_size=64, output_size=128,
                          loss_overrides={"mask_type": "components", "learn_mask": True})

    _feed, target = dataset[0]

    assert not np.all(target[..., 3].numpy() == 255)


# =============================================================================
# MultiDataset
# =============================================================================


class _InputDataset:  # pylint:disable=too-few-public-methods
    """ Minimal dataset stand-in returning a fixed-shape array paired with its index """

    def __init__(self, length: int, value: int) -> None:
        self._length = length
        self._value = value

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        image = np.full((2, 3, 3), self._value, dtype=np.uint8)
        return image, self._value + index


def test_length_is_largest_dataset() -> None:
    """ Size the multi dataset to the length of its largest contained dataset """
    dataset = MultiDataset((_InputDataset(4, 1), _InputDataset(6, 2)))

    assert len(dataset) == 6


def test_stacks_inputs_without_random() -> None:
    """ Stack one item per input dataset at a fixed index when random reads are disabled """
    dataset = MultiDataset((_InputDataset(4, 1), _InputDataset(6, 2)), is_random=False)

    outputs = dataset[0]

    assert len(outputs) == 2
    assert outputs[0].shape == (2, 2, 3, 3)
    assert tuple(outputs[1]) == (1, 2)


def test_stacks_inputs_with_random() -> None:
    """ Stack one item per input dataset when random reads are enabled """
    dataset = MultiDataset((_InputDataset(4, 1), _InputDataset(6, 2)), is_random=True)

    outputs = dataset[0]

    assert len(outputs) == 2
    assert outputs[0].shape == (2, 2, 3, 3)


def test_shuffle_rebuilds_valid_indices() -> None:
    """ Rebuild a full index grid after shuffling that still covers each dataset range """
    dataset = MultiDataset((_InputDataset(4, 1), _InputDataset(6, 2)), is_random=True)

    dataset.shuffle()

    assert isinstance(dataset._indices, np.ndarray)
    assert dataset._indices.shape == (2, 6)
    assert list(np.sort(dataset._indices[1])) == list(range(6))

#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.training.data.collate` """
from __future__ import annotations

import typing as T

import numpy as np
import pytest
import pytest_mock
import torch

from lib.training.data.collate import BatchMeta, Collate, EXTRACT_RATIOS, LandmarkMatcher
from plugins.train.trainer import trainer_config as cfg

from tests.lib.config.helpers import patch_config  # pylint:disable=unused-import  # noqa[F401]

# pylint:disable=protected-access,redefined-outer-name,import-outside-toplevel


# =============================================================================
# Constants
# =============================================================================

MODULE_PREFIX = "lib.training.data.collate"

DEFAULT_AUGMENTATION_CONFIG = {"color_clahe_chance": 0,
                               "color_clahe_max_size": 4,
                               "color_lightness": 0,
                               "color_ab": 0,
                               "rotation_range": 0,
                               "zoom_amount": 0,
                               "shift_range": 0,
                               "flip_chance": 0}

_DEFAULT_COLLATE_PARAMS = {"num_inputs": 2,
                           "batch_size": 2,
                           "input_size": 16,
                           "output_sizes": (32,),
                           "color_order": "bgr",
                           "augment_color": False,
                           "flip": False,
                           "warp": False,
                           "cache_landmarks": False,
                           "landmarks": None}

_BATCH_CASES = ((1,), (3,), (4,), (8,))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def make_collate(patch_config: T.Callable[[object, dict], None],  # noqa: F811
                 ) -> T.Callable[[dict | None], Collate]:
    """ Patch augmentation config and build a Collate instance with overrides """
    def _build(overrides: dict[str, T.Any] | None = None) -> Collate:
        """ Merge the given overrides onto the default collate parameters and build it """
        params = {**_DEFAULT_COLLATE_PARAMS, **(overrides or {})}
        patch_config(cfg.Augmentation, DEFAULT_AUGMENTATION_CONFIG)
        return Collate(**params)

    return _build


@pytest.fixture()
def make_call_data() -> T.Callable[..., list]:
    """ Build a deterministic batch of collated (images, indices) tuples ready for __call__ """
    def _build(num_inputs: int = 2,
               batch_size: int = 2,
               height: int = 8,
               width: int = 8,
               channels: int = 4) -> list[tuple[tuple[np.ndarray, ...], tuple[int, ...]]]:
        """ Return one data entry per sample with a per-input image and its file index """
        rng = np.random.default_rng(0)

        def one_image() -> np.ndarray:
            return (rng.random((height, width, channels)) * 255).astype("uint8")

        data: list[tuple[tuple[np.ndarray, ...], tuple[int, ...]]] = []
        for _ in range(batch_size):
            images = tuple(one_image() for _ in range(num_inputs))
            indices = tuple(range(num_inputs))
            data.append((images, indices))
        return data

    return _build


@pytest.fixture()
def new_landmark_matcher() -> T.Callable[..., LandmarkMatcher]:
    """ Construct a LandmarkMatcher without running file-I/O heavy __init__ """
    def _build(centering: str = "face",
               coverage: float = 0.05,
               y_offset: float = 0.0,
               size: int = 64) -> LandmarkMatcher:
        """ Return a bare instance with the affine setup attributes populated """
        matcher = LandmarkMatcher.__new__(LandmarkMatcher)
        matcher._size = size
        matcher._centering = centering
        matcher._coverage = coverage
        matcher._y_offset = y_offset
        matcher._padding = round(size * (EXTRACT_RATIOS[centering] + coverage - 1)
                                 / (2 * coverage))
        matcher._scale = size - (2 * matcher._padding)
        return matcher

    return _build


# =============================================================================
# BatchMeta - __repr__
# =============================================================================

def test_repr_shows_none_fields() -> None:
    """ Render every mask field as None when none are populated """
    text = repr(BatchMeta())

    assert "BatchMeta(" in text
    assert "mask_face=None" in text
    assert "mask_eye=None" in text
    assert "mask_mouth=None" in text


def test_repr_shows_shapes() -> None:
    """ Render each tensor shape and dtype for populated mask fields """
    tensor = torch.arange(2 * 2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 2, 1, 4, 4)
    text = repr(BatchMeta(mask_face=[tensor]))

    assert "mask_face" in text
    assert "(torch.Size([2, 2, 1, 4, 4]), torch.float32)" in text


# =============================================================================
# BatchMeta - __getitem__ / to
# =============================================================================

def test_getitem_slices_input_index() -> None:
    """ Return a new BatchMeta slicing each mask tensor along the input axis """
    tensor = torch.arange(2 * 2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 2, 1, 4, 4)
    batch_meta = BatchMeta(mask_face=[tensor])

    got = batch_meta[0]

    assert isinstance(got, BatchMeta)
    assert got is not batch_meta
    assert tuple(got.mask_face[0].shape) == (2, 1, 4, 4)


def test_to_returns_self_and_moves() -> None:
    """ Move all tensors to the requested device and return the same instance """
    tensor = torch.zeros((2, 2, 1, 4, 4))
    batch_meta = BatchMeta(mask_face=[tensor])

    got = batch_meta.to("cpu")

    assert got is batch_meta
    assert got.mask_face[0].device.type == "cpu"


# =============================================================================
# Collate - construction / __repr__
# =============================================================================

def test_process_size_uses_largest_size(make_collate: T.Callable[[dict | None], Collate]) -> None:
    """ Derive the processing size as the maximum of inputs and outputs """
    collate = make_collate({"input_size": 16, "output_sizes": (32, 64)})

    assert collate._process_size == 64


def test_resize_targets_flag(make_collate: T.Callable[[dict | None], Collate]) -> None:
    """ Flag target resizing only when output sizes differ from the processing size """
    no_resize = make_collate({"input_size": 8, "output_sizes": (8,)})
    resize = make_collate({"input_size": 8, "output_sizes": (8, 16)})

    assert no_resize._resize_targets is False
    assert resize._resize_targets is True


def test_resize_inputs_flag(make_collate: T.Callable[[dict | None], Collate]) -> None:
    """ Flag input resizing only when the requested input size differs from processing """
    no_resize = make_collate({"input_size": 32, "output_sizes": (32,)})
    resize = make_collate({"input_size": 16, "output_sizes": (32,)})

    assert no_resize._resize_inputs is False
    assert resize._resize_inputs is True


def test_repr_contains_config(make_collate: T.Callable[[dict | None], Collate]) -> None:
    """ Render the collate configuration parameters in the string representation """
    text = repr(make_collate({"input_size": 64, "output_sizes": (128,), "color_order": "rgb"}))

    assert text.startswith("Collate(")
    assert "input_size=64" in text
    assert "output_sizes=(128,)" in text
    assert "color_order=\'rgb\'" in text
    assert "landmarks=None" in text


# =============================================================================
# Collate - _batch_resize
# =============================================================================

@pytest.mark.parametrize("channels", _BATCH_CASES)
def test_resizes_channel_counts(make_collate: T.Callable[[dict | None], Collate],
                                channels: tuple[int, ...]) -> None:
    """ Resize batches of any supported channel count to the requested square size """
    collate = make_collate()
    batch = np.zeros((2, 6, 6, *channels), dtype=np.uint8) + channels[0]

    resized = collate._batch_resize(batch, 10)

    assert resized.dtype == np.uint8
    assert tuple(resized.shape) == (2, 10, 10, *channels)


# =============================================================================
# Collate - _create_targets
# =============================================================================

def test_single_output_single_mask(make_collate: T.Callable[[dict | None], Collate]) -> None:
    """ Build one target and a single face mask when four channels are supplied """
    collate = make_collate({"input_size": 8, "output_sizes": (8,)})
    batch = np.zeros((2 * 2, 8, 8, 4), dtype=np.uint8) + 50

    targets, masks = collate._create_targets(batch)

    assert len(targets) == 1
    assert tuple(targets[0].shape) == (2, 2, 3, 8, 8)
    assert targets[0].dtype == torch.float32
    assert masks.mask_face is not None and len(masks.mask_face) == 1
    assert tuple(masks.mask_face[0].shape) == (2, 2, 1, 8, 8)
    assert masks.mask_eye is None


def test_multiple_outputs_produce_multiple_targets(make_collate: T.Callable[[dict | None], Collate]
                                                   ) -> None:
    """ Build one target per output size and populate every mask for six channels """
    collate = make_collate({"input_size": 8, "output_sizes": (8, 16), "batch_size": 3})
    batch = np.zeros((2 * 3, 8, 8, 6), dtype=np.uint8) + 100

    targets, masks = collate._create_targets(batch)

    assert len(targets) == 2
    assert tuple(targets[0].shape) == (3, 2, 3, 8, 8)
    assert tuple(targets[1].shape) == (3, 2, 3, 16, 16)
    for field in ("mask_face", "mask_eye", "mask_mouth"):
        attr = getattr(masks, field)
        assert attr is not None and len(attr) == 2


def test_all_three_masks_from_six_channels(make_collate: T.Callable[[dict | None], Collate]
                                           ) -> None:
    """ Populate face, eye and mouth masks when six channels are supplied """
    collate = make_collate({"input_size": 8, "output_sizes": (8,)})
    batch = np.zeros((2 * 2, 8, 8, 6), dtype=np.uint8) + 77

    targets, masks = collate._create_targets(batch)

    assert tuple(targets[0].shape) == (2, 2, 3, 8, 8)
    for field in ("mask_face", "mask_eye", "mask_mouth"):
        assert getattr(masks, field) is not None


def test_targets_normalized_to_unit_range(make_collate: T.Callable[[dict | None], Collate]
                                          ) -> None:
    """ Cast target images into the float32 0.0-1.0 range """
    collate = make_collate({"input_size": 8, "output_sizes": (8,)})
    batch = np.zeros((2 * 2, 8, 8, 4), dtype=np.uint8) + 50

    targets, _ = collate._create_targets(batch)

    values = torch.asarray(targets[0]).float()
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0


# =============================================================================
# Collate - _get_landmarks_pairs
# =============================================================================

def test_none_when_warp_disabled(make_collate: T.Callable[[dict | None], Collate]) -> None:
    """ Return no landmark pairs when warping is disabled """
    collate = make_collate({"warp": False, "cache_landmarks": False})

    got = collate._get_landmarks_pairs(np.zeros((2, 2), dtype=np.int64))

    assert got is None


def test_none_without_landmark_object(make_collate: T.Callable[[dict | None], Collate]) -> None:
    """ Return no landmark pairs when warping is enabled but no matcher exists """
    collate = make_collate({"warp": True, "cache_landmarks": False, "landmarks": None})

    got = collate._get_landmarks_pairs(np.zeros((2, 2), dtype=np.int64))

    assert got is None


def test_returns_pair_when_enabled(make_collate: T.Callable[[dict | None], Collate],
                                   mocker: pytest_mock.MockerFixture) -> None:
    """ Forward the indices to the matcher and return its landmark pairs """
    pairs = np.zeros((8, 2, 68, 2), dtype=np.float32)
    landmarks = mocker.MagicMock()
    landmarks.get_close_landmarks.return_value = pairs
    collate = make_collate({"warp": True, "cache_landmarks": True, "landmarks": landmarks})

    got = collate._get_landmarks_pairs(np.zeros((2, 2), dtype=np.int64))

    assert got is pairs


# =============================================================================
# Collate - __call__
# =============================================================================

def test_call_returns_expected_shapes(make_collate: T.Callable[[dict | None], Collate],
                                      make_call_data: T.Callable[..., list]) -> None:
    """ Feed, target and mask outputs with the documented tensor shapes for bgr """
    collate = make_collate({"input_size": 16, "output_sizes": (32,)})
    data = make_call_data(num_inputs=2, batch_size=2, channels=4)

    inputs, targets, masks = collate(data)

    assert len(inputs) == 2
    for feed in inputs:
        assert isinstance(feed, torch.Tensor)
        assert feed.dtype == torch.float32
        assert tuple(feed.shape) == (2, 3, 16, 16)
    assert len(targets) == 1
    assert tuple(targets[0].shape) == (2, 2, 3, 8, 8)
    assert isinstance(masks, BatchMeta)


def test_rgb_swaps_channels_vs_bgr(make_collate: T.Callable[[dict | None], Collate],
                                   make_call_data: T.Callable[..., list]) -> None:
    """ Rearrange channels for rgb inputs while leaving bgr ordering untouched """
    bgr = make_collate({"color_order": "bgr", "input_size": 16, "output_sizes": (32,)})
    rgb = make_collate({"color_order": "rgb", "input_size": 16, "output_sizes": (32,)})
    data = make_call_data()

    bgr_in, _, _ = bgr(data)
    rgb_in, _, _ = rgb(data)

    assert not np.array_equal(np.asarray(bgr_in[0]), np.asarray(rgb_in[0]))


def test_warp_runs_without_landmark_object(make_collate: T.Callable[[dict | None], Collate],
                                           make_call_data: T.Callable[..., list]) -> None:
    """ Apply standard warping and return correctly shaped inputs when no matcher exists """
    collate = make_collate({"warp": True, "input_size": 16, "output_sizes": (32,)})
    data = make_call_data()

    inputs, targets, _ = collate(data)

    assert len(inputs) == 2
    for feed in inputs:
        assert tuple(feed.shape) == (2, 3, 16, 16)
    assert isinstance(targets[0], torch.Tensor)


# =============================================================================
# LandmarkMatcher - construction / __repr__
# =============================================================================

def test_requires_two_folders() -> None:
    """ Fail when Warp-to-Landmarks is configured with anything other than two sides """
    with pytest.raises(AssertionError):
        LandmarkMatcher(folders=["/a"], size=64, centering="face", coverage=0.05, y_offset=0.0)


def test_padding_and_scale_from_config(mocker: pytest_mock.MockerFixture) -> None:
    """ Derive padding and scale from the size, centering ratio and coverage """
    mocker.patch.object(LandmarkMatcher, "_load_landmarks",
                        return_value=(np.zeros((2, 68, 2)), np.zeros((3, 68, 2))))

    matcher = LandmarkMatcher(folders=["/a", "/b"], size=64, centering="face",
                              coverage=0.05, y_offset=0.0)

    expected_padding = round(64 * (EXTRACT_RATIOS["face"] + 0.05 - 1) / (2 * 0.05))
    assert matcher._padding == expected_padding
    assert matcher._scale == 64 - (2 * expected_padding)


def test_num_choices_clamped(mocker: pytest_mock.MockerFixture) -> None:
    """ Reduce the number of choices so it never exceeds the smallest side minus one """
    mocker.patch.object(LandmarkMatcher, "_load_landmarks",
                        return_value=(np.zeros((5, 68, 2)), np.zeros((4, 68, 2))))

    matcher = LandmarkMatcher(folders=["/a", "/b"], size=64, centering="face",
                              coverage=0.05, y_offset=0.0, num_choices=100)

    assert matcher._num_choices == 3


def test_repr_shows_configured_parameters(mocker: pytest_mock.MockerFixture) -> None:
    """ Render the configured folders, size, centering, coverage and offset but not internals """
    mocker.patch.object(LandmarkMatcher, "_load_landmarks",
                        return_value=(np.zeros((2, 68, 2)), np.zeros((3, 68, 2))))

    matcher = LandmarkMatcher(folders=["/a", "/b"], size=64, centering="face",
                              coverage=0.05, y_offset=0.0)

    text = repr(matcher)

    assert "folders=" in text
    assert "size=64" in text
    assert "centering='face'" in text
    assert "coverage=0.05" in text


# =============================================================================
# LandmarkMatcher - _landmarks_from_header
# =============================================================================

def test_valid_header_returns_68_points() -> None:
    """ Extract the frame-space landmarks from a well-formed face image metadata dict """
    matcher = LandmarkMatcher.__new__(LandmarkMatcher)
    meta = {"itxt": {"alignments": {"landmarks_xy": np.zeros((68, 2), dtype=np.float32)}}}

    got = matcher._landmarks_from_header(meta, "face.png")

    assert got.shape == (68, 2)
    assert got.dtype == np.float32


def test_header_without_itxt_raises() -> None:
    """ Raise when the face image metadata has no itxt section """
    from lib.utils import FaceswapError
    matcher = LandmarkMatcher.__new__(LandmarkMatcher)

    with pytest.raises(FaceswapError):
        matcher._landmarks_from_header({"no_itxt": True}, "face.png")


def test_header_without_alignments_raises() -> None:
    """ Raise when the itxt section has no alignments subsection """
    from lib.utils import FaceswapError
    matcher = LandmarkMatcher.__new__(LandmarkMatcher)

    with pytest.raises(FaceswapError):
        matcher._landmarks_from_header({"itxt": {}}, "face.png")


def test_header_wrong_landmark_count_raises() -> None:
    """ Raise when the metadata holds landmarks other than 68 points """
    from lib.utils import FaceswapError
    matcher = LandmarkMatcher.__new__(LandmarkMatcher)
    meta = {"itxt": {"alignments": {"landmarks_xy": np.zeros((51, 2), dtype=np.float32)}}}

    with pytest.raises(FaceswapError):
        matcher._landmarks_from_header(meta, "face.png")


# =============================================================================
# LandmarkMatcher - _align_points / _get_closest_indices / get_close_landmarks
# =============================================================================

def test_align_points_shape_and_dtype(new_landmark_matcher: T.Callable[..., LandmarkMatcher]
                                      ) -> None:
    """ Normalize and align landmark points to the model input size preserving shape and type """
    matcher = new_landmark_matcher()
    points = np.random.default_rng(0).random((3, 68, 2)).astype("float32") * 64

    aligned = matcher._align_points(points)

    assert aligned.shape == (3, 68, 2)
    assert aligned.dtype == np.float32


def test_y_offset_shifts_landmarks(new_landmark_matcher: T.Callable[..., LandmarkMatcher]) -> None:
    """ Apply the configured vertical offset so landmarks differ from an unoffset build """
    points = np.random.default_rng(0).random((3, 68, 2)).astype("float32") * 64
    reference = new_landmark_matcher(y_offset=0.0)._align_points(points)
    shifted = new_landmark_matcher(y_offset=5.0)._align_points(points)

    assert not np.array_equal(reference, shifted)


def test_closest_indices_shapes(new_landmark_matcher: T.Callable[..., LandmarkMatcher]) -> None:
    """ Return the nearest opposite-side candidates for every landmark within the choice limit """
    matcher = new_landmark_matcher()
    landmarks_a = np.random.default_rng(1).random((5, 68, 2)).astype("float32")
    landmarks_b = np.random.default_rng(2).random((4, 68, 2)).astype("float32")
    matcher._landmarks = (landmarks_a, landmarks_b)
    matcher._num_choices = 3
    matcher._closest_indices = matcher._get_closest_indices()

    matches_a, matches_b = matcher._closest_indices[0], matcher._closest_indices[1]

    assert matches_a.shape == (5, 3)
    assert matches_b.shape == (4, 3)
    assert np.all(matches_a >= 0) and np.all(matches_a < landmarks_b.shape[0])
    assert np.all(matches_b >= 0) and np.all(matches_b < landmarks_a.shape[0])


def test_get_close_landmarks_shape(new_landmark_matcher: T.Callable[..., LandmarkMatcher]) -> None:
    """ Return stacked source/destination landmark pairs shaped for augmentation """
    matcher = new_landmark_matcher()
    landmarks_a = np.random.default_rng(1).random((5, 68, 2)).astype("float32")
    landmarks_b = np.random.default_rng(2).random((4, 68, 2)).astype("float32")
    matcher._landmarks = (landmarks_a, landmarks_b)
    matcher._num_choices = 3
    matcher._closest_indices = matcher._get_closest_indices()

    result = matcher.get_close_landmarks(np.array([[0, 1], [2, 3]]))

    assert result.dtype == np.float32
    assert result.shape[1:] == (2, 68, 2)

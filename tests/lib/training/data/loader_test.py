#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.training.data.loader` """
import logging
import os
from pathlib import Path

import pytest
import pytest_mock
import torch
from torch.utils import data as tch_data
from torch.utils.data import DataLoader

from lib.training.data.collate import BatchMeta
from lib.training.data.loader import PreviewLoader, TrainLoader, validate_faceswap_folders

# pylint:disable=protected-access,redefined-outer-name


# =============================================================================
# Constants
# =============================================================================

MODULE_PREFIX = "lib.training.data.loader"

_VALID_PATHS = [f"/fake/side/{i}.png" for i in range(300)]  # >=250 -> no warning
_LOW_PATHS = [f"/fake/side/{i}.png" for i in range(100)]     # >=25,<250 -> warns
_TINY_PATHS = [f"/fake/side/{i}.png" for i in range(10)]     # <25 -> exits


# =============================================================================
# Fixtures
# =============================================================================

class _FakeDataset:  # pylint:disable=too-few-public-methods
    """ Minimal dataset stand-in exposing a fixed length for loader tests """
    def __init__(self, size: int = 16) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size


class _FakeMultiDataset:  # pylint:disable=too-few-public-methods
    """ Stand-in for MultiDataset exposing a fixed length and no-op shuffle """
    def __init__(self) -> None:
        self._size = 16

    def __len__(self) -> int:
        return self._size

    def shuffle(self) -> None:
        """ No-op stand-in so loader tests avoid touching real indices """


# =============================================================================
# Validate faceswap folders
# =============================================================================

def test_valid_folders_returned(tmp_path: Path,
                                mocker: pytest_mock.MockerFixture) -> None:
    """ Return the input folders unchanged when every side is valid """
    folder_a = str(tmp_path / "a")
    folder_b = str(tmp_path / "b")
    os.makedirs(folder_a)
    os.makedirs(folder_b)
    mocker.patch(f"{MODULE_PREFIX}.get_image_paths", return_value=_VALID_PATHS)
    mocker.patch(f"{MODULE_PREFIX}.validate_faceswap_image", return_value=True)
    folders = [folder_a, folder_b]

    assert validate_faceswap_folders(folders) is folders


def test_missing_folder_exits(tmp_path: Path, mocker: pytest_mock.MockerFixture) -> None:
    """ Exit when a configured folder does not exist """
    missing = str(tmp_path / "does_not_exist")
    mocker.patch(f"{MODULE_PREFIX}.get_image_paths", return_value=_VALID_PATHS)

    with pytest.raises(SystemExit):
        validate_faceswap_folders([missing])


def test_no_images_exits(tmp_path: Path, mocker: pytest_mock.MockerFixture) -> None:
    """ Exit when a folder contains no images """
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    mocker.patch(f"{MODULE_PREFIX}.get_image_paths", return_value=[])

    with pytest.raises(SystemExit):
        validate_faceswap_folders([empty])


def test_invalid_face_exits(tmp_path: Path, mocker: pytest_mock.MockerFixture) -> None:
    """ Exit when the first image fails faceswap validation """
    folder = str(tmp_path / "a")
    os.makedirs(folder)
    mocker.patch(f"{MODULE_PREFIX}.get_image_paths", return_value=_VALID_PATHS)
    mocker.patch(f"{MODULE_PREFIX}.validate_faceswap_image", return_value=False)

    with pytest.raises(SystemExit):
        validate_faceswap_folders([folder])


def test_few_images_exits(tmp_path: Path, mocker: pytest_mock.MockerFixture) -> None:
    """ Exit when a side has fewer than 25 images """
    folder = str(tmp_path / "a")
    os.makedirs(folder)
    mocker.patch(f"{MODULE_PREFIX}.get_image_paths", return_value=_TINY_PATHS)
    mocker.patch(f"{MODULE_PREFIX}.validate_faceswap_image", return_value=True)

    with pytest.raises(SystemExit):
        validate_faceswap_folders([folder])


def test_low_count_warns(tmp_path: Path,
                         caplog: pytest.LogCaptureFixture,
                         mocker: pytest_mock.MockerFixture) -> None:
    """ Warn (rather than exit) when a side has between 25 and 250 images """
    folder = str(tmp_path / "a")
    os.makedirs(folder)
    mocker.patch(f"{MODULE_PREFIX}.get_image_paths", return_value=_LOW_PATHS)
    mocker.patch(f"{MODULE_PREFIX}.validate_faceswap_image", return_value=True)

    with caplog.at_level(logging.WARNING, logger="lib.training.data.loader"):
        result = validate_faceswap_folders([folder])

    assert result == [folder]
    assert any("fewer than 250 images" in rec.message for rec in caplog.records)


# =============================================================================
# TrainLoader - __repr__
# =============================================================================

def test_repr_lists_public_params() -> None:
    """ Render the configured public parameters and hide augmentation config """
    loader = TrainLoader.__new__(TrainLoader)
    loader._aug_config = {"augment_color": True,
                          "flip": False,
                          "warp": False,
                          "cache_landmarks": False}
    loader._folders = ["/a", "/b"]
    loader._batch_size = 4
    loader._input_size = 256
    loader._output_sizes = (256,)
    loader._color_order = "bgr"
    loader._sampler = tch_data.RandomSampler

    text = repr(loader)

    assert text.startswith("TrainLoader(")
    assert "folders=['/a', '/b']" in text
    assert "batch_size=4" in text
    assert "input_size=256" in text
    assert "output_sizes=(256,)" in text
    assert "color_order='bgr'" in text
    assert "_aug_config" not in text


# =============================================================================
# TrainLoader - get_loader
# =============================================================================

def test_get_loader_builds_dataloader(tmp_path: Path,
                                      mocker: pytest_mock.MockerFixture) -> None:
    """ Build and return a DataLoader assembled from the per-side datasets """
    folders = [str(tmp_path / "a"), str(tmp_path / "b")]
    mocker.patch(f"{MODULE_PREFIX}.validate_faceswap_folders", return_value=folders)
    mocker.patch(f"{MODULE_PREFIX}.TrainSet", return_value=_FakeDataset(16))
    mocker.patch(f"{MODULE_PREFIX}.MultiDataset", return_value=_FakeMultiDataset())
    collate_mock = mocker.patch(f"{MODULE_PREFIX}.Collate", return_value=mocker.MagicMock())
    mocker.patch("plugins.train.trainer.trainer_config.Loader.num_processes", lambda: 0)
    mocker.patch("plugins.train.trainer.trainer_config.Loader.pre_fetch", lambda: None)

    loader_obj = TrainLoader(folders=folders,
                             batch_size=4,
                             input_size=256,
                             output_sizes=(256,),
                             color_order="bgr",
                             augment_color=False,
                             flip=False,
                             warp=False,
                             cache_landmarks=False)

    assert isinstance(loader_obj._loader, DataLoader)
    assert loader_obj._loader.batch_size == 4
    assert loader_obj._loader.collate_fn is collate_mock.return_value


def test_get_loader_uses_passed_sampler(mocker: pytest_mock.MockerFixture) -> None:
    """ Use the caller-supplied sampler instead of the random default """
    folders = ["/a", "/b"]
    mocker.patch(f"{MODULE_PREFIX}.validate_faceswap_folders", return_value=folders)
    mocker.patch(f"{MODULE_PREFIX}.TrainSet", return_value=_FakeDataset(16))
    mocker.patch(f"{MODULE_PREFIX}.MultiDataset", return_value=_FakeMultiDataset())
    mocker.patch(f"{MODULE_PREFIX}.Collate", return_value=mocker.MagicMock())
    mocker.patch("plugins.train.trainer.trainer_config.Loader.num_processes", lambda: 0)
    mocker.patch("plugins.train.trainer.trainer_config.Loader.pre_fetch", lambda: None)

    loader_obj = TrainLoader(folders=folders,
                             batch_size=4,
                             input_size=256,
                             output_sizes=(256,),
                             color_order="bgr",
                             augment_color=False,
                             flip=False,
                             warp=False,
                             cache_landmarks=False,
                             sampler=tch_data.SequentialSampler)

    assert isinstance(loader_obj._loader.sampler, tch_data.SequentialSampler)


# =============================================================================
# TrainLoader - __next__
# =============================================================================

def test_next_returns_batch_when_mask_disabled() -> None:
    """ Return the collated batch unchanged when learn-mask is disabled """
    loader = TrainLoader.__new__(TrainLoader)
    loader._learn_mask = False
    loader._epoch = 0
    inputs = [torch.zeros(1, 3, 8, 8)]
    targets = [torch.zeros(1, 3, 8, 8, 3)]
    meta = BatchMeta()
    loader._iterator = iter([(inputs, targets, meta)])

    got_inputs, got_targets, got_meta = next(loader)

    assert got_inputs is inputs
    assert got_targets is targets
    assert got_meta is meta


def test_next_appends_face_mask_when_learned() -> None:
    """ Append the face mask as an extra target when learn-mask is enabled """
    loader = TrainLoader.__new__(TrainLoader)
    loader._learn_mask = True
    loader._epoch = 0
    inputs = [torch.zeros(1, 3, 8, 8)]
    targets = [torch.zeros(1, 3, 8, 8, 3)]
    face_mask = torch.zeros(1, 2, 4, 4)
    meta = BatchMeta(mask_face=[face_mask])
    loader._iterator = iter([(inputs, targets, meta)])

    _, got_targets, _ = next(loader)

    assert len(got_targets) == 2
    assert got_targets[-1] is face_mask


def test_next_wraps_at_epoch_boundary(mocker: pytest_mock.MockerFixture) -> None:
    """ Refetch the batch and bump the epoch when a side is exhausted """
    loader = TrainLoader.__new__(TrainLoader)
    loader._learn_mask = False
    loader._epoch = 0
    fresh_inputs = [torch.zeros(1, 3, 8, 8)]
    fresh_targets = [torch.zeros(1, 3, 8, 8, 3)]
    fresh_meta = BatchMeta()
    mocker.patch(f"{MODULE_PREFIX}.MultiDataset")
    loader._loader = mocker.MagicMock()
    loader._loader.__iter__.return_value = iter([(fresh_inputs, fresh_targets, fresh_meta)])
    loader._iterator = iter([])

    got_inputs, _, _ = next(loader)

    assert got_inputs is fresh_inputs
    assert loader._epoch == 1


# =============================================================================
# PreviewLoader - construction / repr
# =============================================================================

def test_input_folders_property(tmp_path: Path, mocker: pytest_mock.MockerFixture) -> None:
    """ Expose the configured input folders through a property """
    folders = [str(tmp_path / "a"), str(tmp_path / "b")]
    mocker.patch(f"{MODULE_PREFIX}.PreviewSet", return_value=_FakeDataset(16))

    loader_obj = PreviewLoader(input_size=256,
                               output_size=256,
                               color_order="bgr",
                               input_folders=folders,
                               batch_size=4)

    assert loader_obj.input_folders == folders


def test_repr_lists_all_params() -> None:
    """ Render every configured parameter including the number of samples """
    loader = PreviewLoader.__new__(PreviewLoader)
    loader._input_size = 256
    loader._output_size = 256
    loader._color_order = "bgr"
    loader._input_folders = ["/a", "/b"]
    loader._batch_size = 4
    loader._sampler = tch_data.RandomSampler
    loader._num_samples = 0

    text = repr(loader)

    assert text.startswith("PreviewLoader(")
    assert "input_size=256" in text
    assert "output_size=256" in text
    assert "color_order='bgr'" in text
    assert "input_folders=['/a', '/b']" in text
    assert "batch_size=4" in text
    assert "num_samples=0" in text


def test_repr_hides_unknown_params() -> None:
    """ Only render the configured parameters, hiding any other attributes """
    loader = PreviewLoader.__new__(PreviewLoader)
    loader._input_size = 256
    loader._output_size = 256
    loader._color_order = "bgr"
    loader._input_folders = ["/a"]
    loader._batch_size = 4
    loader._sampler = tch_data.RandomSampler
    loader._num_samples = 0
    loader._extra_hidden = "hidden"

    text = repr(loader)

    assert "_extra_hidden" not in text


# =============================================================================
# PreviewLoader - get_loader
# =============================================================================

def test_preview_builds_dataloader(tmp_path: Path, mocker: pytest_mock.MockerFixture) -> None:
    """ Build and return a DataLoader assembled from the per-side preview datasets """
    folders = [str(tmp_path / "a"), str(tmp_path / "b")]
    mocker.patch(f"{MODULE_PREFIX}.PreviewSet", return_value=_FakeDataset(16))

    loader_obj = PreviewLoader(input_size=256,
                               output_size=256,
                               color_order="bgr",
                               input_folders=folders,
                               batch_size=4)

    assert isinstance(loader_obj._loader, DataLoader)
    assert loader_obj._loader.batch_size == 4


# =============================================================================
# PreviewLoader - _items_from_loader / __next__
# =============================================================================

def test_items_returned_as_is() -> None:
    """ Return the collated feed and targets from the iterator unchanged """
    loader = PreviewLoader.__new__(PreviewLoader)
    feed = torch.zeros(2, 4, 8, 8)
    targets = torch.zeros(2, 4, 8, 12)
    loader._iterator = iter([(feed, targets)])

    got_feed, got_targets = loader._items_from_loader()

    assert got_feed is feed
    assert got_targets is targets


def test_items_wrap_at_boundary(mocker: pytest_mock.MockerFixture) -> None:
    """ Refetch from the loader when the iterator is exhausted """
    loader = PreviewLoader.__new__(PreviewLoader)
    fresh_feed = torch.zeros(2, 4, 8, 8)
    fresh_targets = torch.zeros(2, 4, 8, 12)
    loader._loader = mocker.MagicMock()
    loader._loader.__iter__.return_value = iter([(fresh_feed, fresh_targets)])
    loader._iterator = iter([])

    got_feed, got_targets = loader._items_from_loader()

    assert got_feed is fresh_feed
    assert got_targets is fresh_targets


def test_next_swaps_axes() -> None:
    """ Swap the batch and side axes of both feed and targets on each next call """
    loader = PreviewLoader.__new__(PreviewLoader)
    feed = torch.zeros(2, 4, 8)
    targets = torch.zeros(2, 4, 8, 1)
    loader._iterator = iter([(feed, targets)])

    got_inputs, got_targets = next(loader)

    assert got_inputs.shape == feed.swapaxes(0, 1).shape
    assert got_targets.shape == targets.swapaxes(0, 1).shape

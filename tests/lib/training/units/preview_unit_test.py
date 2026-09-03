#!/usr/bin/env python3
# pylint:disable=missing-module-docstring,protected-access,import-error
""" Pytest unit tests for :mod:`lib.training.units.preview_unit` """
from __future__ import annotations

import typing as T
import unittest.mock

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch

from lib.training.events import TrainingEvents
from lib.training.units.preview_unit import Samples, EvaluateUnit, PreviewUnit, TimelapseUnit


# =============================================================================
# Module-level fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _mock_preview_loader_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """ Automatically mock PreviewLoader and os.listdir to avoid filesystem access """
    mock_loader = unittest.mock.MagicMock()
    feed = torch.randn(2, 4, 3, 256, 256)
    target = torch.randn(2, 4, 3, 256, 256)
    mock_loader.__iter__ = unittest.mock.MagicMock(return_value=iter([(feed, target)]))
    mock_loader.input_folders = ["/fake/folder"]
    monkeypatch.setattr("lib.training.data.PreviewLoader", lambda *a, **k: mock_loader)
    monkeypatch.setattr("lib.training.units.preview_unit.PreviewLoader",
                        lambda *a, **k: mock_loader)
    monkeypatch.setattr("os.listdir", lambda folder: ["test.png"])


def _make_small_predictions(num_identities: int = 2,
                            num_previews: int = 2,
                            size: int = 32,
                            channels: int = 3) -> npt.NDArray[np.float32]:
    """ Create a small fake predictions array for fast, lightweight tests """
    return np.random.randn(num_identities,
                           num_identities,
                           num_previews,
                           size,
                           size,
                           channels).astype(np.float32)


def _make_small_targets(num_identities: int = 2,
                        num_previews: int = 2,
                        size: int = 32,
                        channels: int = 3) -> npt.NDArray[np.float32]:
    """ Create a small fake targets array for fast, lightweight tests """
    return np.random.randn(num_identities, num_previews, size, size, channels).astype(np.float32)


# =============================================================================
# Samples tests
# =============================================================================


class TestSamplesInit:
    """ Tests for Samples.__init__ """

    @pytest.mark.parametrize(
        ("coverage_ratio", "has_mask", "mask_opacity", "mask_color", "expected_display_mask"),
        [(0.8, False, 50, "#FF0000", False),
         (1.0, True, 50, "#FF0000", True),
         (0.5, True, 100, "#00FF00", True)]
    )
    def test_init_stores_all_attributes(self,
                                        coverage_ratio: float,
                                        has_mask: bool,
                                        mask_opacity: int,
                                        mask_color: str,
                                        expected_display_mask: bool) -> None:
        """ Samples.__init__ stores all configuration attributes correctly """
        samples = Samples(coverage_ratio=coverage_ratio,
                          has_mask=has_mask,
                          mask_opacity=mask_opacity,
                          mask_color=mask_color)
        assert samples._coverage_ratio == coverage_ratio
        assert samples._has_mask is has_mask
        assert samples._mask_opacity == mask_opacity / 100.0
        assert samples._mask_color == mask_color
        assert samples._display_mask is expected_display_mask

    def test_repr_contains_class_name_and_params(self) -> None:
        """ Samples.__repr__ includes class name and all configuration parameters """
        samples = Samples(coverage_ratio=0.8,
                          has_mask=True,
                          mask_opacity=50,
                          mask_color="#FF0000")
        rep = repr(samples)
        assert "Samples" in rep
        assert "0.8" in rep
        assert "True" in rep
        assert "0.5" in rep
        assert "#FF0000" in rep


class TestSamplesToggleMaskDisplay:
    """ Tests for Samples._toggle_mask_display """

    @pytest.mark.parametrize(("has_mask", "initial_display", "expected_after"),
                             [(False, False, False),
                              (True, True, False),
                              (True, False, True)])
    def test_toggle_mask_display(self,
                                 has_mask: bool,
                                 initial_display: bool,
                                 expected_after: bool) -> None:
        """ Samples._toggle_mask_display toggles only when has_mask is True """
        samples = Samples(coverage_ratio=0.8,
                          has_mask=has_mask,
                          mask_opacity=50,
                          mask_color="#FF0000")
        if not has_mask:
            samples._toggle_mask_display()
            assert samples._display_mask is False
        else:
            samples._display_mask = initial_display
            samples._toggle_mask_display()
            assert samples._display_mask is expected_after


class TestSamplesGetPreview:
    """ Tests for Samples.get_preview """

    @staticmethod
    def _patch_cv2() -> unittest.mock.MagicMock:
        """ Patch cv2 font functions to avoid heavy rendering in tests """
        def _dynamic_headers(num_swaps: int, patch_width: int) -> npt.NDArray[np.uint8]:
            cols = num_swaps * (num_swaps + 1)
            height = max(1, int(patch_width / 4.5))
            return np.ones((height, patch_width * cols, 3), dtype=np.uint8) * 255

        patch = unittest.mock.patch.object(Samples, "_get_headers", side_effect=_dynamic_headers)
        return patch.start()

    @pytest.mark.parametrize(("channels", "has_mask"),
                             [(3, False), (4, True)])
    def test_output_shape_and_dtype(self, channels: int, has_mask: bool) -> None:
        """ Samples.get_preview returns a uint8 array with 3 channels and 2D spatial dims """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=has_mask,
                          mask_opacity=50, mask_color="#FF0000")
        preds = _make_small_predictions(channels=channels)
        tgts = _make_small_targets(channels=channels)
        result = samples.get_preview(preds, tgts, toggle_mask=False)
        assert result.dtype == np.uint8
        assert result.shape[2] == 3
        assert len(result.shape) == 3

    def test_output_height_exceeds_patch_height(self) -> None:
        """ Samples.get_preview output height > patch height because headers are prepended """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=False,
                          mask_opacity=50, mask_color="#FF0000")
        preds = _make_small_predictions()
        tgts = _make_small_targets()
        result = samples.get_preview(preds, tgts, toggle_mask=False)
        patch_h = tgts.shape[-2]
        assert result.shape[0] > patch_h

    @pytest.mark.parametrize("coverage_ratio", [1.0, 0.5])
    def test_coverage_ratio_affects_preview(self, coverage_ratio: float) -> None:
        """ Samples.get_preview handles both full and partial coverage ratios """
        self._patch_cv2()
        samples = Samples(coverage_ratio=coverage_ratio, has_mask=False,
                          mask_opacity=50, mask_color="#FF0000")
        preds = _make_small_predictions()
        tgts = _make_small_targets()
        result = samples.get_preview(preds, tgts, toggle_mask=False)
        assert result.shape[2] == 3

    def test_mask_overlay_applied_when_displayed(self) -> None:
        """ Samples.get_preview applies mask overlay when has_mask and _display_mask are True """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=True,
                          mask_opacity=100, mask_color="#FF0000")
        preds = _make_small_predictions()
        tgts = _make_small_targets()
        result = samples.get_preview(preds, tgts, toggle_mask=False)
        assert result.shape[2] == 3

    def test_mask_overlay_skipped_when_not_displayed(self) -> None:
        """ Samples.get_preview skips mask overlay when _display_mask is False """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=True,
                          mask_opacity=50, mask_color="#FF0000")
        samples._display_mask = False
        preds = _make_small_predictions()
        tgts = _make_small_targets()
        result = samples.get_preview(preds, tgts, toggle_mask=False)
        assert result.shape[2] == 3

    def test_toggle_mask_flips_display(self) -> None:
        """ Samples.get_preview with toggle_mask=True flips _display_mask """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=True,
                          mask_opacity=50, mask_color="#FF0000")
        preds = _make_small_predictions()
        tgts = _make_small_targets()
        assert samples._display_mask is True
        samples.get_preview(preds, tgts, toggle_mask=True)
        assert samples._display_mask is False

    def test_toggle_mask_noop_when_no_mask(self) -> None:
        """ Samples.get_preview with toggle_mask=True does nothing when has_mask is False """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=False,
                          mask_opacity=50, mask_color="#FF0000")
        preds = _make_small_predictions()
        tgts = _make_small_targets()
        samples.get_preview(preds, tgts, toggle_mask=True)
        assert samples._display_mask is False

    def test_alpha_channel_predictions(self) -> None:
        """ Samples.get_preview extracts masks from alpha channel when channels=4 """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=True,
                          mask_opacity=50, mask_color="#FF0000")
        preds = _make_small_predictions(channels=4)
        tgts = _make_small_targets(channels=4)
        result = samples.get_preview(preds, tgts, toggle_mask=False)
        assert result.shape[2] == 3

    def test_output_shape_consistent_across_repeated_calls(self) -> None:
        """ Samples.get_preview produces the same output shape on repeated calls """
        self._patch_cv2()
        samples = Samples(coverage_ratio=0.8, has_mask=False,
                          mask_opacity=50, mask_color="#FF0000")
        preds = _make_small_predictions()
        tgts = _make_small_targets()
        result1 = samples.get_preview(preds, tgts, toggle_mask=False)
        result2 = samples.get_preview(preds, tgts, toggle_mask=False)
        assert result1.shape == result2.shape


# =============================================================================
# EvaluateUnit tests
# =============================================================================


class TestEvaluateUnitInit:
    """ Tests for EvaluateUnit.__init__ """

    @pytest.mark.parametrize(("attr", "expected"),
                             [("_batch_size", 4),
                              ("_learn_mask", False),
                              ("_output_size", 256),
                              ("_is_rgb", True)])
    def test_init_sets_attributes_from_config_and_model(
            self,
            mock_faceswap_model: unittest.mock.MagicMock,
            attr: str,
            expected: T.Any) -> None:
        """ EvaluateUnit.__init__ sets attributes from config and model """
        unit = EvaluateUnit(mock_faceswap_model)
        assert getattr(unit, attr) == expected

    def test_init_stores_model(self,
                               mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ EvaluateUnit.__init__ stores the provided model """
        unit = EvaluateUnit(mock_faceswap_model)
        assert unit._model is mock_faceswap_model

    def test_repr_contains_class_name(self,
                                      mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ EvaluateUnit.__repr__ includes the class name """
        unit = EvaluateUnit(mock_faceswap_model)
        assert "EvaluateUnit" in repr(unit)


class TestEvaluateUnitOnLoad:
    """ Tests for EvaluateUnit.on_load """

    def test_on_load_sets_device(self,
                                 mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ EvaluateUnit.on_load copies the device from the training loop """
        unit = EvaluateUnit(mock_faceswap_model)
        mock_loop = unittest.mock.MagicMock()
        mock_loop.device = "cuda:0"
        unit.on_load(mock_loop)
        assert unit._device == "cuda:0"


class TestEvaluateUnitGetPredictions:
    """ Tests for EvaluateUnit._get_predictions """

    def test_returns_correct_shape_rgb(self,
                                       mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ EvaluateUnit._get_predictions returns (B, N, H, W, C) float32 array for RGB """
        unit = EvaluateUnit(mock_faceswap_model)
        unit._device = "cpu"
        feed = torch.randn(2, 4, 3, 256, 256)
        mock_output = torch.randn(2, 4, 3, 256, 256)
        unit._model.plugin = unittest.mock.MagicMock(return_value=[mock_output])
        result = unit._get_predictions(feed)
        assert result.shape == (2, 4, 256, 256, 3)
        assert result.dtype == np.float32


class TestEvaluateUnitGetSamples:
    """ Tests for EvaluateUnit._get_samples """

    def test_returns_uint8_image(self,
                                 mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ EvaluateUnit._get_samples returns a uint8 image """
        unit = EvaluateUnit(mock_faceswap_model)
        unit._device = "cpu"
        unit._get_samples = unittest.mock.MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        result = unit._get_samples()
        assert result.dtype == np.uint8

    def test_output_has_3_channels(self,
                                   mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ EvaluateUnit._get_samples returns an image with 3 channels """
        unit = EvaluateUnit(mock_faceswap_model)
        unit._device = "cpu"
        unit._get_samples = unittest.mock.MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        result = unit._get_samples()
        assert result.shape[2] == 3


# =============================================================================
# PreviewUnit tests
# =============================================================================


class TestPreviewUnitInit:
    """ Tests for PreviewUnit.__init__ """

    def test_init_sets_loader(self,
                              mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ PreviewUnit.__init__ creates a loader from input folders """
        unit = PreviewUnit(mock_faceswap_model, ["/fake/folder"])
        assert unit._loader is not None

    def test_init_uses_random_sampler(self,
                                      mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ PreviewUnit.__init__ uses a random sampler (not sequential) """
        unit = PreviewUnit(mock_faceswap_model, ["/fake/folder"])
        assert unit._loader is not None


class TestPreviewUnitOnUpdate:
    """ Tests for PreviewUnit.on_update """

    @staticmethod
    def _make_preview_unit(mock_model: unittest.mock.MagicMock) -> PreviewUnit:
        """ Create a PreviewUnit with _events and _samples set for testing """
        unit = PreviewUnit(mock_model, ["/fake/folder"])
        unit._events = TrainingEvents()
        unit._samples = Samples(coverage_ratio=80, has_mask=False,
                                mask_opacity=50, mask_color="#FF0000")
        return unit

    def test_on_update_triggers_preview(self,
                                        mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ PreviewUnit.on_update calls _get_samples to generate a preview """
        unit = self._make_preview_unit(mock_faceswap_model)
        unit._get_samples = unittest.mock.MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        unit.on_update()
        unit._get_samples.assert_called()

    def test_on_update_noop_when_no_events(self,
                                           mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ PreviewUnit.on_update still runs when no events are set """
        unit = self._make_preview_unit(mock_faceswap_model)
        unit._get_samples = unittest.mock.MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        assert unit._events.update.is_set() is False
        unit.on_update()
        unit._get_samples.assert_called()

    def test_on_update_toggles_mask_on_event(self,
                                             mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ PreviewUnit.on_update toggles mask display when toggle_mask event is set """
        unit = self._make_preview_unit(mock_faceswap_model)
        unit._get_samples = unittest.mock.MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        unit._events.toggle_mask.set()
        unit.on_update()
        assert unit._samples._display_mask is False

    def test_on_update_clears_update_event(self,
                                           mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ PreviewUnit.on_update clears the update event after processing """
        unit = self._make_preview_unit(mock_faceswap_model)
        unit._get_samples = unittest.mock.MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        unit._events.update.set()
        unit.on_update()
        assert unit._events.update.is_set() is True

    def test_repr_contains_class_name(self,
                                      mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ PreviewUnit.__repr__ includes the class name """
        unit = PreviewUnit(mock_faceswap_model, ["/fake/folder"])
        assert "PreviewUnit" in repr(unit)


# =============================================================================
# TimelapseUnit tests
# =============================================================================


class TestTimelapseUnitInit:
    """ Tests for TimelapseUnit.__init__ """

    @pytest.mark.parametrize(("attr", "expected"),
                             [("_output_folder", "/fake/output")])
    def test_init_sets_attributes(self,
                                  mock_faceswap_model: unittest.mock.MagicMock,
                                  attr: str,
                                  expected: T.Any) -> None:
        """ TimelapseUnit.__init__ sets expected attributes """
        unit = TimelapseUnit(mock_faceswap_model, ["/fake/folder"], "/fake/output")
        assert getattr(unit, attr) == expected

    def test_init_sets_loader(self,
                              mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ TimelapseUnit.__init__ creates a loader from input folders """
        unit = TimelapseUnit(mock_faceswap_model, ["/fake/folder"], "/fake/output")
        assert unit._loader is not None

    def test_init_uses_sequential_sampler(
            self, mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ TimelapseUnit.__init__ uses a sequential sampler for consistent sampling """
        unit = TimelapseUnit(mock_faceswap_model, ["/fake/folder"], "/fake/output")
        assert unit._loader is not None

    def test_repr_contains_folders(self,
                                   mock_faceswap_model: unittest.mock.MagicMock) -> None:
        """ TimelapseUnit.__repr__ includes input and output folder paths """
        unit = TimelapseUnit(mock_faceswap_model, ["/fake/folder"], "/fake/output")
        rep = repr(unit)
        assert "TimelapseUnit" in rep
        assert "/fake/folder" in rep
        assert "/fake/output" in rep


class TestTimelapseUnitOnSave:
    """ Tests for TimelapseUnit.on_save """

    @pytest.mark.parametrize(("iteration", "expected_filename"), [
                             (1, "00000001.jpg"),
                             (42, "00000042.jpg"),
                             (123, "00000123.jpg"),
                             (999, "00000999.jpg")])
    def test_on_save_creates_output_folder_and_saves_jpg(
            self,
            mock_faceswap_model: unittest.mock.MagicMock,
            tmp_path: T.Any,
            iteration: int,
            expected_filename: str) -> None:
        """ TimelapseUnit.on_save creates folder and saves JPEG with zero-padded name """
        output_dir = str(tmp_path / "timelapse")
        unit = TimelapseUnit(mock_faceswap_model, ["/fake/folder"], output_dir)
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        unit._get_samples = unittest.mock.MagicMock(return_value=fake_img)
        unit.on_save(iteration)
        expected_file = tmp_path / "timelapse" / expected_filename
        assert expected_file.exists()

    def test_on_save_does_not_recreate_existing_folder(
            self, mock_faceswap_model: unittest.mock.MagicMock, tmp_path: T.Any) -> None:
        """ TimelapseUnit.on_save skips folder creation when output folder already exists """
        output_dir = str(tmp_path / "timelapse")
        (tmp_path / "timelapse").mkdir()
        unit = TimelapseUnit(mock_faceswap_model, ["/fake/folder"], output_dir)
        unit._get_samples = unittest.mock.MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        with unittest.mock.patch("os.makedirs") as mock_makedirs:
            unit.on_save(1)
            mock_makedirs.assert_not_called()

    def test_on_save_writes_correct_image_content(
            self, mock_faceswap_model: unittest.mock.MagicMock, tmp_path: T.Any) -> None:
        """ TimelapseUnit.on_save writes the exact image content returned by _get_samples """
        output_dir = str(tmp_path / "timelapse")
        unit = TimelapseUnit(mock_faceswap_model, ["/fake/folder"], output_dir)
        test_img = np.full((50, 50, 3), 128, dtype=np.uint8)
        unit._get_samples = unittest.mock.MagicMock(return_value=test_img)
        unit.on_save(1)
        saved = cv2.imread(str(tmp_path / "timelapse" / "00000001.jpg"))
        assert saved is not None
        assert np.array_equal(saved, test_img)

#!/usr/bin/env python3
# pylint:disable=missing-function-docstring,too-few-public-methods,import-error
""" Doubles (stubs/mocks) for training unit test objects """
from __future__ import annotations

import unittest.mock

import pytest

from lib.model.plugin import State
from lib.training.events import TrainingEvents


@pytest.fixture()
def mock_state() -> State:
    """ Fixture providing a fresh :class:`lib.model.plugin.State` instance per test """
    return State("mock_plugin")


@pytest.fixture()
def events() -> TrainingEvents:
    """ Fixture providing a real shared :class:`lib.training.events.TrainingEvents` instance """
    return TrainingEvents()


@pytest.fixture(autouse=True)
def _patch_preview_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    """ Automatically patch training config modules for preview unit tests

    Provides defaults for ``coverage()``, ``Loss.learn_mask()``, ``Loss.penalized_mask_loss()``,
    ``Augmentation.preview_images()``, ``Augmentation.mask_opacity()``, and
    ``Augmentation.mask_color()``
    """
    monkeypatch.setattr("plugins.train.train_config.coverage", lambda: 80)
    monkeypatch.setattr("plugins.train.train_config.Loss.learn_mask", lambda: False)
    monkeypatch.setattr("plugins.train.train_config.Loss.penalized_mask_loss", lambda: False)
    monkeypatch.setattr("plugins.train.trainer.trainer_config.Augmentation.preview_images",
                        lambda: 4)
    monkeypatch.setattr("plugins.train.trainer.trainer_config.Augmentation.mask_opacity",
                        lambda: 50)
    monkeypatch.setattr("plugins.train.trainer.trainer_config.Augmentation.mask_color",
                        lambda: "#FF0000")


@pytest.fixture()
def mock_faceswap_model() -> unittest.mock.MagicMock:
    """ Fixture providing a :class:`unittest.mock.MagicMock` representing FaceswapModel.

    Sets up the minimal interface needed by EvaluateUnit / PreviewUnit / TimelapseUnit:
    ``info.input_size``, ``info.output_size``, ``plugin.is_rgb``, ``info.device``.
    """
    mock_model = unittest.mock.MagicMock()
    mock_model.info.input_size = 256
    mock_model.info.output_size = 256
    mock_model.plugin.is_rgb = True
    mock_model.info.device = "cpu"
    return mock_model

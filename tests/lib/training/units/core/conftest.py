#!/usr/bin/env python3
# pylint:disable=missing-function-docstring,too-few-public-methods,import-error
""" Doubles (stubs/mocks) for Core unit test objects """
from __future__ import annotations

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

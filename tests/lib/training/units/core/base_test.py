""" Unit tests for lib.training.units.core.base """
# pylint:disable=too-few-public-methods,unnecessary-pass,missing-function-docstring
from typing import Any
import pytest

from lib.training.units.core.base import TrainingUnit  # pylint:disable=import-error


# =============================================================================
# Constants
# =============================================================================


# Maps a method name to the has_* capability property that reports whether it is overridden
_METHOD_TO_PROPERTY = {"on_load": "has_load",
                       "on_start": "has_start",
                       "step": "has_step",
                       "on_save": "has_save",
                       "on_update": "has_update",
                       "on_end": "has_end",
                       "load_state_dict": "has_state_dict",
                       "state_dict": "has_state_dict"}


# =============================================================================
# Fixtures
# =============================================================================


class MockTrainingUnit(TrainingUnit):
    """ A mock unit that overrides nothing """
    pass


class MockLoadUnit(TrainingUnit):
    """ A mock unit that overrides on_load """
    def on_load(self, loop: Any) -> None:
        pass


class MockStartUnit(TrainingUnit):
    """ A mock unit that overrides on_start """
    def on_start(self) -> None:
        pass


class MockStepUnit(TrainingUnit):
    """ A mock unit that overrides step """
    def step(self, iteration: int) -> None:
        pass


class MockSaveUnit(TrainingUnit):
    """ A mock unit that overrides on_save """
    def on_save(self, iteration: int) -> None:
        pass


class MockUpdateUnit(TrainingUnit):
    """ A mock unit that overrides on_update """
    def on_update(self) -> None:
        pass


class MockEndUnit(TrainingUnit):
    """ A mock unit that overrides on_end """
    def on_end(self) -> None:
        pass


class MockStateDictUnit(TrainingUnit):
    """ A mock unit that overrides both state_dict and load_state_dict """
    def state_dict(self) -> dict[str, Any] | None:
        return {"mock": "data"}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class MockPartialStateDictUnit(TrainingUnit):
    """ A mock unit that only overrides state_dict """
    def state_dict(self) -> dict[str, Any] | None:
        return {"mock": "data"}


class MockOnlyLoadStateDictUnit(TrainingUnit):
    """ A mock unit that only overrides load_state_dict """
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class UnitSuffix(TrainingUnit):
    """ A mock unit whose class name ends in 'Unit' """
    pass


class SimpleUnit(TrainingUnit):
    """ A mock unit whose class name ends in 'Unit' with a different prefix """
    pass


class DataLogger(TrainingUnit):
    """ A mock unit whose class name does not end in 'Unit' """
    pass


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize(("unit", "expected_log_name"),
                         [(UnitSuffix, "[UnitSuffix]"),
                          (SimpleUnit, "[Simple]"),
                          (DataLogger, "[DataLogger]")])
def test_log_name_formatting(unit: type[TrainingUnit], expected_log_name: str) -> None:
    """ A unit's log_name is '[ClassName]' with a trailing 'Unit' stripped """
    assert unit().log_name == expected_log_name


@pytest.mark.parametrize(("mock_class", "method", "expected"),
                         # A unit that overrides nothing reports no capabilities
                         [(MockTrainingUnit, "on_load", False),
                          # Each unit reports True only for the method it overrides
                          (MockLoadUnit, "on_load", True),
                          (MockStartUnit, "on_start", True),
                          (MockStepUnit, "step", True),
                          (MockSaveUnit, "on_save", True),
                          (MockUpdateUnit, "on_update", True),
                          (MockEndUnit, "on_end", True),
                          (MockStateDictUnit, "state_dict", True),
                          # has_state_dict is False when only one persistence method is overridden
                          (MockPartialStateDictUnit, "state_dict", False),
                          (MockOnlyLoadStateDictUnit, "load_state_dict", False)])
def test_capability_detection(mock_class: type[TrainingUnit], method: str, expected: bool) -> None:
    """ has_* reports True only for methods overridden by the child class """
    unit = mock_class()
    property_name = _METHOD_TO_PROPERTY[method]
    assert getattr(unit, property_name) is expected


def test_state_dict_capability() -> None:
    """ Verify that has_state_dict is only True if both methods are overridden """
    # Both overridden
    assert MockStateDictUnit().has_state_dict is True

    # Only state_dict overridden
    assert MockPartialStateDictUnit().has_state_dict is False

    # Only load_state_dict overridden
    assert MockOnlyLoadStateDictUnit().has_state_dict is False

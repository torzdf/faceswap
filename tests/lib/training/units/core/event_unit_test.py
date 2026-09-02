#!/usr/bin python3
# pylint:disable=missing-class-docstring,unused-import,protected-access,import-error
""" Pytest unit tests for :mod:`lib.training.units.core.event_unit` """
from __future__ import annotations

import pytest

from lib.training.events import TrainingEvents
from lib.training.units.core.event_unit import EventUnit
from .core_doubles import fixture_events  # noqa:F401


# =============================================================================
# Initialization and Configuration
# =============================================================================


class TestEventUnitInit:
    def test_init_stores_shared_events_reference(self, events: TrainingEvents) -> None:
        """ Constructor wires the injected events object into the unit """
        unit = EventUnit(events)
        assert unit._events is events


class TestEventUnitRepr:
    def test_repr_surfaces_class_name_and_events(self, events: TrainingEvents) -> None:
        """ The string representation exposes the class name and injected events object """
        unit = EventUnit(events)
        text = repr(unit)
        assert "EventUnit" in text
        assert "events=" in text
        assert f"{events!r}" in text


# =============================================================================
# Step Processing
# =============================================================================


class TestEventUnitStep:
    @pytest.mark.parametrize("iteration", [-2, -1, 0, 5])
    def test_step_update_on_mask_toggled_no_pending(self,
                                                    caplog: pytest.LogCaptureFixture,
                                                    events: TrainingEvents,
                                                    iteration: int) -> None:
        """ A pending mask toggle with no update queued triggers an update """
        with caplog.at_level("CRITICAL"):
            events.toggle_mask.set()
            unit = EventUnit(events)
            unit.step(iteration)
            assert events.update.is_set() is True

    @pytest.mark.parametrize("iteration", [-2, -1, 0, 5])
    def test_step_no_action_when_mask_not_toggled(self,
                                                  caplog: pytest.LogCaptureFixture,
                                                  events: TrainingEvents,
                                                  iteration: int) -> None:
        """ With no mask toggle pending, step leaves the update event clear """
        with caplog.at_level("CRITICAL"):
            unit = EventUnit(events)
            unit.step(iteration)
            assert events.update.is_set() is False

    @pytest.mark.parametrize("iteration", [-2, -1, 0, 5])
    def test_step_no_trigger_on_update_pending(self,
                                               caplog: pytest.LogCaptureFixture,
                                               events: TrainingEvents,
                                               iteration: int) -> None:
        """ If an update is already pending, step must not raise or alter state """
        with caplog.at_level("CRITICAL"):
            events.toggle_mask.set()
            unit = EventUnit(events)
            events.update.set()
            unit.step(iteration)
            assert events.update.is_set() is True

    def test_step_triggers_once_per_pending_request(self,
                                                    caplog: pytest.LogCaptureFixture,
                                                    events: TrainingEvents) -> None:
        """ Each consumed pending request triggers exactly one update while mask stays toggled """
        with caplog.at_level("CRITICAL"):
            events.toggle_mask.set()
            unit = EventUnit(events)
            unit.step(1)
            assert events.update.is_set() is True
            events.update.clear()
            unit.step(2)
            assert events.update.is_set() is True

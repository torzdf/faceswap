#!/usr/bin/env python3
# pylint:disable=unused-import,import-error
""" Pytest unit tests for :mod:`lib.training.units.core.state_unit` """
from __future__ import annotations

import pytest

from lib.model.plugin import State
from lib.training.units.core.state_unit import StateUnit


# =============================================================================
# Initialization and Configuration
# =============================================================================


class TestStateUnitInit:
    """ Tests for StateUnit construction. """
    def test_init_starts_counters_at_zero(self, mock_state: State) -> None:
        """ Construction records zero iterations and zero session iterations """
        unit = StateUnit(mock_state, 32)
        assert unit.iteration == 0
        assert unit.session_iteration == 0

    def test_init_repr(self, mock_state: State) -> None:
        """ The string representation exposes the class name and applied batch size """
        unit = StateUnit(mock_state, 32)
        text = repr(unit)
        assert "StateUnit" in text
        assert "batch_size=32" in text


# =============================================================================
# Step Processing
# =============================================================================


class TestStateUnitStep:
    """ Tests for step(). """
    @pytest.mark.parametrize("steps", [1, 2, 5])
    def test_step_advances_counters(self, mock_state: State, steps: int) -> None:
        """ Each call to step() advances both counters by one in training mode """
        unit = StateUnit(mock_state, 32)
        for _ in range(steps):
            unit.step(0)
        assert unit.iteration == steps
        assert unit.session_iteration == steps

    @pytest.mark.parametrize("steps", [1, 2])
    def test_step_noop_in_pre_train(self, mock_state: State, steps: int) -> None:
        """ step() does not advance counters while the state is in pre-training mode """
        unit = StateUnit(mock_state, 32)
        assert unit.iteration == 0
        mock_state.set_pre_training()
        for _ in range(steps):
            unit.step(1)
        # In pre-train mode stepping is skipped and the total counter stays at -1;
        # only session iteration remains unchanged.
        assert unit.iteration == -1
        assert unit.session_iteration == 0

    def test_step_resumes_after_pre_train(self, mock_state: State) -> None:
        """ Stepping resumes once the state transitions out of pre-training mode """
        unit = StateUnit(mock_state, 32)
        assert unit.iteration == 0
        # Exit pre-train mode and resume counting from zero.
        mock_state.set_pre_training()
        unit.step(1)
        mock_state.set_training()
        unit.step(1)
        assert unit.iteration == 1
        assert unit.session_iteration == 1


# =============================================================================
# Session Iteration Property
# =============================================================================


class TestStateUnitSessionIteration:
    """ Tests for the session_iteration property """

    def test_session_iteration_tracks_current_session(self, mock_state: State) -> None:
        """ session_iteration reflects the current session's iteration count """
        unit = StateUnit(mock_state, 32)
        unit.step(1)
        unit.step(2)
        assert unit.session_iteration == 2

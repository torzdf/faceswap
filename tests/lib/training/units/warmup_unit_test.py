#!/usr/bin/env python3
# pylint:disable=missing-class-docstring,missing-function-docstring,protected-access,import-error
""" Pytest unit tests for :mod:`lib.training.units.warmup_unit` """
from __future__ import annotations

import pytest
import torch
from torch import nn

from lib.training.units.warmup_unit import WarmupScheduler, WarmupUnit


def _make_optimizer(lr: float = 0.01) -> torch.optim.Optimizer:
    """ Helper to create a minimal real PyTorch optimizer for testing """
    param = nn.Parameter(torch.randn(1, requires_grad=True))
    return torch.optim.SGD([param], lr=lr)


# =============================================================================
# WarmupScheduler — Initialization
# =============================================================================


class TestWarmupSchedulerInit:
    """ Tests for WarmupScheduler construction """
    def test_init_stores_steps(self) -> None:
        optimizer = _make_optimizer()
        scheduler = WarmupScheduler(optimizer, steps=100)
        assert scheduler.steps == 100


# =============================================================================
# WarmupScheduler — get_lr()
# =============================================================================


class TestWarmupSchedulerGetLR:
    """ Tests for WarmupScheduler.get_lr() """
    @pytest.mark.parametrize("steps_after_init,expected_factor", [
        (0, 0.0),
        (1, 0.25),
        (2, 0.5),
        (3, 0.75),
        (4, 1.0),
    ])
    def test_during_warmup_linear_progression(
        self,
        steps_after_init: int,
        expected_factor: float,
    ) -> None:
        """ get_lr() scales each parameter group's LR linearly from 0 to base_lr during warmup """
        optimizer = _make_optimizer(lr=0.001)
        scheduler = WarmupScheduler(optimizer, steps=4)
        # PyTorch LRScheduler.__init__ calls step() once (last_epoch -1 -> 0),
        # so steps_after_init additional calls advance further.
        for _ in range(steps_after_init):
            scheduler.step()
        lrs = scheduler.get_lr()
        assert lrs == [0.001 * expected_factor]

    def test_at_boundary_returns_base_lr(self) -> None:
        """ When last_epoch >= steps, get_lr() returns base_lrs (no overshoot) """
        optimizer = _make_optimizer(lr=0.01)
        scheduler = WarmupScheduler(optimizer, steps=5)
        # init does 1 step (last_epoch -1 -> 0), then 6 more = 7 total
        for _ in range(7):
            scheduler.step()
        lrs = scheduler.get_lr()
        assert lrs == [0.01]

    def test_after_warmup_returns_base_lr(self) -> None:
        """ Once past warmup, get_lr() keeps returning base_lrs """
        optimizer = _make_optimizer(lr=0.05)
        scheduler = WarmupScheduler(optimizer, steps=3)
        for _ in range(10):
            scheduler.step()
        lrs = scheduler.get_lr()
        assert lrs == [0.05]

    def test_multiple_parameter_groups_each_scale_linearly(self) -> None:
        """ Each param group scales from its own base_lr, independently """
        params = [
            nn.Parameter(torch.randn(1, requires_grad=True)),
            nn.Parameter(torch.randn(1, requires_grad=True)),
        ]
        optimizer = torch.optim.SGD([{'params': [params[0]], 'lr': 0.01}, {
                                    'params': [params[1]], 'lr': 0.02}])
        scheduler = WarmupScheduler(optimizer, steps=4)
        # init does 1 step, then 1 more = last_epoch 1, factor 0.25
        scheduler.step()
        lrs = scheduler.get_lr()
        assert lrs == [0.01 * 0.25, 0.02 * 0.25]


# =============================================================================
# WarmupUnit — Initialization
# =============================================================================


class TestWarmupUnitInit:
    """ Tests for WarmupUnit construction """
    def test_init_stores_warmup_steps_and_reporting_points(self) -> None:
        unit = WarmupUnit(warmup_steps=100)
        assert unit._warmup_steps == 100
        assert unit._reporting_points == [
            int(100 * i / 10) for i in range(11)
        ]

    def test_init_starts_iteration_at_zero(self) -> None:
        unit = WarmupUnit(warmup_steps=50)
        assert unit._iteration == 0

    def test_init_zero_warmup_steps(self) -> None:
        unit = WarmupUnit(warmup_steps=0)
        assert unit._warmup_steps == 0
        assert unit._reporting_points == [0] * 11


# =============================================================================
# WarmupUnit — __repr__
# =============================================================================


class TestWarmupUnitRepr:
    """ Tests for WarmupUnit.__repr__ """
    def test_repr_returns_class_name_and_warmup_steps(self) -> None:
        unit = WarmupUnit(warmup_steps=200)
        text = repr(unit)
        assert "WarmupUnit" in text
        assert "warmup_steps=200" in text


# =============================================================================
# WarmupUnit — _fmt
# =============================================================================


class TestWarmupUnitFmt:
    """ Tests for WarmupUnit._fmt """
    @pytest.mark.parametrize("value,expected", [
        (0.001, "1.0e-03"),
        (0.01, "1.0e-02"),
        (0.1, "1.0e-01"),
        (0.0, "0.0e+00"),
        (1.0, "1.0e+00"),
        (0.0005, "5.0e-04"),
    ])
    def test_fmt_formats_various_floats_in_scientific_notation(
        self,
        value: float,
        expected: str,
    ) -> None:
        assert WarmupUnit._fmt(value) == expected


# =============================================================================
# WarmupUnit — on_load
# =============================================================================


class TestWarmupUnitOnLoad:
    """ Tests for WarmupUnit.on_load """
    def test_stores_optimizer_reference_from_loop(self) -> None:
        mock_optimizer = _make_optimizer()
        mock_optimizer_unit = type('MockOptimizerUnit', (), {'optimizer': mock_optimizer})()
        mock_loop = type('MockLoop', (), {'optimizer_unit': mock_optimizer_unit})()

        unit = WarmupUnit(warmup_steps=10)
        unit.on_load(mock_loop)
        assert unit._optimizer is mock_optimizer


# =============================================================================
# WarmupUnit — on_start
# =============================================================================


class TestWarmupUnitOnStart:
    """ Tests for WarmupUnit.on_start """
    def test_creates_scheduler_with_optimizer_and_steps(self) -> None:
        mock_optimizer = _make_optimizer()
        unit = WarmupUnit(warmup_steps=50)
        unit._optimizer = mock_optimizer
        unit.on_start()
        assert isinstance(unit._scheduler, WarmupScheduler)
        assert unit._scheduler.steps == 50

    def test_scheduler_receives_optimizer(self) -> None:
        mock_optimizer = _make_optimizer()
        unit = WarmupUnit(warmup_steps=30)
        unit._optimizer = mock_optimizer
        unit.on_start()
        assert hasattr(unit._scheduler, "base_lrs")


# =============================================================================
# WarmupUnit — step()
# =============================================================================


class TestWarmupUnitStep:
    """ Tests for WarmupUnit.step() """
    def test_advances_iteration_and_calls_scheduler_during_warmup(self) -> None:
        mock_optimizer = _make_optimizer()
        unit = WarmupUnit(warmup_steps=10)
        unit._optimizer = mock_optimizer
        unit.on_start()
        assert unit._iteration == 0

        unit.step(1)
        assert unit._iteration == 1

        unit.step(2)
        assert unit._iteration == 2

    def test_noop_when_iteration_already_past_warmup(self) -> None:
        mock_optimizer = _make_optimizer()
        unit = WarmupUnit(warmup_steps=3)
        unit._optimizer = mock_optimizer
        unit.on_start()

        unit._iteration = 5
        unit.step(10)
        assert unit._iteration == 5

    def test_noop_when_pre_training_iteration_lt_1(self) -> None:
        mock_optimizer = _make_optimizer()
        unit = WarmupUnit(warmup_steps=10)
        unit._optimizer = mock_optimizer
        unit.on_start()

        unit.step(0)
        assert unit._iteration == 0

        unit.step(-1)
        assert unit._iteration == 0

    def test_step_continues_at_boundary_iteration_equals_warmup(self) -> None:
        """ When _iteration == warmup_steps, step still proceeds (condition is >, not >=) """
        mock_optimizer = _make_optimizer()
        unit = WarmupUnit(warmup_steps=5)
        unit._optimizer = mock_optimizer
        unit.on_start()

        for i in range(1, 6):
            unit.step(i)
        assert unit._iteration == 5

        # _iteration == warmup_steps, so step continues (not a no-op)
        unit.step(6)
        assert unit._iteration == 6


# =============================================================================
# WarmupUnit — _report_progress
# =============================================================================


class TestWarmupUnitReportProgress:
    """ Tests for WarmupUnit._report_progress """
    @pytest.fixture()
    def _warmup_unit_with_scheduler(self) -> WarmupUnit:
        mock_optimizer = _make_optimizer(lr=0.01)
        unit = WarmupUnit(warmup_steps=10)
        unit._optimizer = mock_optimizer
        unit.on_start()
        return unit

    def test_logs_start_message_at_step_1(
        self,
        _warmup_unit_with_scheduler: WarmupUnit,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ Step 1 logs a start message with LR info """
        with caplog.at_level("INFO"):
            _warmup_unit_with_scheduler.step(1)
        assert any("Start:" in m for m in caplog.messages)

    def test_logs_final_message_at_warmup_steps(
        self,
        _warmup_unit_with_scheduler: WarmupUnit,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ Final step logs the final learning rate """
        with caplog.at_level("INFO"):
            for i in range(1, 11):
                _warmup_unit_with_scheduler.step(i)
        assert any("Final Learning Rate:" in m for m in caplog.messages)

    def test_logs_intermediate_progress_at_reporting_points(
        self,
        _warmup_unit_with_scheduler: WarmupUnit,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ Intermediate reporting points log progress percentage """
        with caplog.at_level("INFO"):
            for i in range(1, 11):
                _warmup_unit_with_scheduler.step(i)
        assert any("%" in m for m in caplog.messages)

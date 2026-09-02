#!/usr/bin/env python3
# pylint:disable=protected-access, redefined-outer-name, unused-argument
""" Unit tests for LossUnit - loss calculation and monitoring during training

Tests the observable behavior of the LossUnit class: configuration echoed in the
string representation, running-average computation, save-time contribution
reporting, NaN protection, console loss printing, and the training-unit
capability contract. Follows black-box testing: asserts only on observable
outputs (current_average, console output, log messages, exceptions) and never
on private state or internal call counts
"""
# pylint:disable=import-error
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from lib.training.units.core.loss_unit import LossUnit
from lib.utils import FaceswapError
from tests.lib.training.loss.batch_loss_mock import BatchLossMock

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def two_component_loss() -> BatchLossMock:
    """ Mock BatchLoss with two loss components for CPU testing """
    return BatchLossMock(unweighted=[{"loss1": torch.tensor(1.0), "loss2": torch.tensor(2.0)}],
                         weighted=[{"loss1": torch.tensor(1.5), "loss2": torch.tensor(3.0)}],
                         mask=None)


@pytest.fixture
def single_component_loss() -> BatchLossMock:
    """ Mock BatchLoss with a single loss component """
    return BatchLossMock(unweighted=[{"loss1": torch.tensor(1.0)}],
                         weighted=[{"loss1": torch.tensor(1.5)}],
                         mask=None)


@pytest.fixture
def loss_unit(two_component_loss: BatchLossMock) -> LossUnit:
    """ LossUnit instance operating on the CPU with two loss components """
    return LossUnit(nan_protection=False,
                    current_loss=[two_component_loss],
                    device=torch.device("cpu"))


@pytest.fixture
def loss_unit_nan(two_component_loss: BatchLossMock) -> LossUnit:
    """ LossUnit instance with NaN protection enabled """
    return LossUnit(nan_protection=True,
                    current_loss=[two_component_loss],
                    device=torch.device("cpu"))


@pytest.fixture
def loss_unit_single(single_component_loss: BatchLossMock) -> LossUnit:
    """ LossUnit instance operating on the CPU with a single loss component """
    return LossUnit(nan_protection=False,
                    current_loss=[single_component_loss],
                    device=torch.device("cpu"))


# =============================================================================
# Initialization and State
# =============================================================================


def test_repr_reports_configuration(loss_unit: LossUnit) -> None:
    """ LossUnit.__repr__ exposes nan_protection and device for debugging """
    repr_str = repr(loss_unit)
    assert "LossUnit" in repr_str
    assert "nan_protection=False" in repr_str
    assert "cpu" in repr_str


def test_initial_current_average_is_zero(loss_unit: LossUnit) -> None:
    """ Before any step, the reported average loss is zero """
    assert loss_unit.current_average.item() == 0.0


def test_capability_flags_report_step_and_save(loss_unit: LossUnit) -> None:
    """ LossUnit advertises its step and save capabilities to the training loop """
    assert loss_unit.has_step is True
    assert loss_unit.has_save is True


# =============================================================================
# Average Management
# =============================================================================


def test_step_updates_running_average(loss_unit: LossUnit) -> None:
    """ A single step followed by save reports the batch weighted average """
    loss_unit.step(iteration=1)
    loss_unit.on_save(iteration=2)
    assert loss_unit.current_average.item() == pytest.approx(4.5)


def test_running_average_accumulates_batch_totals() -> None:
    """ Repeated steps accumulate the running average of batch weighted totals """
    losses = [BatchLossMock(unweighted=[{"loss1": torch.tensor(1.0)}],
                            weighted=[{"loss1": torch.tensor(2.0)}],
                            mask=None),
              BatchLossMock(unweighted=[{"loss1": torch.tensor(2.0)}],
                            weighted=[{"loss1": torch.tensor(4.0)}],
                            mask=None),
              BatchLossMock(unweighted=[{"loss1": torch.tensor(3.0)}],
                            weighted=[{"loss1": torch.tensor(6.0)}],
                            mask=None)]
    unit = LossUnit(nan_protection=False, current_loss=losses, device=torch.device("cpu"))
    for index, _ in enumerate(losses, start=1):
        unit.step(iteration=index)
    unit.on_save(iteration=100)
    assert unit.current_average.item() == pytest.approx(12.0)


def test_on_save_resets_running_average() -> None:
    """ A save stores the current average before resetting for the next window """
    losses = [BatchLossMock(unweighted=[{"loss1": torch.tensor(1.0)}],
                            weighted=[{"loss1": torch.tensor(1.0)}],
                            mask=None),
              BatchLossMock(unweighted=[{"loss1": torch.tensor(1.0)}],
                            weighted=[{"loss1": torch.tensor(1.0)}],
                            mask=None),
              BatchLossMock(unweighted=[{"loss1": torch.tensor(1.0)}],
                            weighted=[{"loss1": torch.tensor(9.0)}],
                            mask=None)]
    unit = LossUnit(nan_protection=False, current_loss=losses, device=torch.device("cpu"))
    unit.step(iteration=1)
    unit.on_save(iteration=2)
    unit.step(iteration=3)
    unit.on_save(iteration=4)
    assert unit.current_average.item() == pytest.approx(11.0)


def test_on_save_reports_all_groups(loss_unit: LossUnit, caplog: pytest.LogCaptureFixture) -> None:
    """ on_save logs both weighted and unweighted contribution ratios """
    loss_unit.step(iteration=1)
    with caplog.at_level(logging.INFO):
        loss_unit.on_save(iteration=1000)
    assert "Weighted" in caplog.text
    assert "Unweighted" in caplog.text


def test_single_component_on_save_reports_full_ratios(loss_unit_single: LossUnit,
                                                      caplog: pytest.LogCaptureFixture) -> None:
    """ With a single component both weighted and unweighted ratios are 100% """
    loss_unit_single.step(iteration=1)
    with caplog.at_level(logging.INFO):
        loss_unit_single.on_save(iteration=2)
    assert "100.0%" in caplog.text


# =============================================================================
# NaN Protection
# =============================================================================


def test_nan_without_protection_is_ignored() -> None:
    """ With protection disabled NaN losses are processed without error """
    nan_loss = BatchLossMock(unweighted=[{"loss1": torch.tensor(float("nan"))}],
                             weighted=[{"loss1": torch.tensor(float("nan"))}],
                             mask=None)
    unit = LossUnit(nan_protection=False, current_loss=[nan_loss], device=torch.device("cpu"))
    unit.step(iteration=1)  # must not raise


def test_nan_with_protection_raises(loss_unit_nan: LossUnit) -> None:
    """ With protection enabled NaN losses terminate training via FaceswapError """
    nan_loss = BatchLossMock(unweighted=[{"loss1": torch.tensor(float("nan"))}],
                             weighted=[{"loss1": torch.tensor(float("nan"))}],
                             mask=None)
    loss_unit_nan._loss = [nan_loss]
    with pytest.raises(FaceswapError, match="NaN"):
        loss_unit_nan.step(iteration=1)


def test_valid_loss_passes_nan_check(loss_unit_nan: LossUnit) -> None:
    """ Valid losses pass the NaN check when protection is enabled """
    good_loss = BatchLossMock(unweighted=[{"loss1": torch.tensor(1.0)}],
                              weighted=[{"loss1": torch.tensor(1.5)}],
                              mask=None)
    unit = LossUnit(nan_protection=True, current_loss=[good_loss], device=torch.device("cpu"))
    unit.step(iteration=1)  # must not raise
    unit.on_save(iteration=2)
    assert unit.current_average.item() == pytest.approx(1.5)


# =============================================================================
# Step Processing
# =============================================================================


def test_step_skips_pretraining(loss_unit: LossUnit) -> None:
    """ Negative iterations (pre-training) are skipped without updating averages """
    loss_unit.step(iteration=-1)
    assert loss_unit.current_average.item() == 0.0


def test_step_processes_normal_iteration(loss_unit: LossUnit) -> None:
    """ A normal step updates the running average """
    loss_unit.step(iteration=100)
    loss_unit.on_save(iteration=101)
    assert loss_unit.current_average.item() == pytest.approx(4.5)


# =============================================================================
# Loss Printing
# =============================================================================


def test_step_prints_formatted_loss(loss_unit: LossUnit, capsys: pytest.CaptureFixture) -> None:
    """ step emits a formatted, timestamped loss line to the console """
    loss_unit.step(iteration=12345)
    captured = capsys.readouterr()
    assert "Loss A:" in captured.out
    assert "12345" in captured.out


# =============================================================================
# Save Operations
# =============================================================================


def test_on_save_updates_current_average(loss_unit: LossUnit) -> None:
    """ on_save sets current_average to the accumulated weighted average """
    loss_unit.step(iteration=1)
    loss_unit.on_save(iteration=1000)
    assert loss_unit.current_average.item() == pytest.approx(4.5)


def test_on_save_skips_when_no_averages(loss_unit: LossUnit) -> None:
    """ on_save is a no-op before any step has updated the averages """
    loss_unit.on_save(iteration=1)
    assert loss_unit.current_average.item() == 0.0


# =============================================================================
# Type Contract
# =============================================================================


def test_current_average_is_float32_array(loss_unit: LossUnit) -> None:
    """ current_average reports a float32 numpy array """
    assert isinstance(loss_unit.current_average, np.ndarray)
    assert loss_unit.current_average.dtype == np.float32


# =============================================================================
# Known Limitations
# =============================================================================


def test_on_save_with_single_component(loss_unit_single: LossUnit,
                                       caplog: pytest.LogCaptureFixture) -> None:
    """ on_save works with a single loss component without error """
    loss_unit_single.step(iteration=1)
    with caplog.at_level(logging.INFO):
        loss_unit_single.on_save(iteration=2)
    assert "Ratios" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

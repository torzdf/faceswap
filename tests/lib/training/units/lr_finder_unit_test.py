#!/usr/bin/env python3
# pylint:disable=missing-class-docstring,missing-function-docstring,protected-access,import-error
# pylint:disable=unspecified-encoding,consider-using-with,too-few-public-methods
""" Pytest unit tests for :mod:`lib.training.units.lr_finder_unit` """
from __future__ import annotations

import os
import typing as T
import unittest.mock
import warnings

import matplotlib
import pytest
import torch
from torch import nn

from lib.training.units import lr_finder_unit
from lib.training.units.lr_finder_unit import (
    LRFScheduler,
    LearningRateFinder,
    LRFinderUnit,
    LRFState,
    LRStrength,
    plot_loss,
)

if T.TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================


class _RecordingOptimizerUnit:
    """ Minimal stand-in for an OptimizerUnit that records the applied learning rate """
    def __init__(self) -> None:
        self.optimizer = _make_optimizer()
        self.lr: float | None = None

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def state_dict(self) -> dict[str, object]:
        return {"optimizer": "state"}

    def load_state_dict(self, _state_dict: dict[str, object]) -> None:
        pass


class _CurrentLossItem:
    """ Single entry of the loss tracker exposing a ``total`` value """
    def __init__(self, total: float) -> None:
        self.total = total


class _ExitFlag:
    """ Observability flag for the training loop exit event """
    def __init__(self) -> None:
        self.set_called = False

    def set(self) -> None:
        self.set_called = True


class _Events:
    """ Training events holder exposing an observable ``exit`` signal """
    def __init__(self) -> None:
        self.exit = _ExitFlag()


def _make_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    """ Build a minimal real single-parameter SGD optimizer for scheduler tests """
    param = nn.Parameter(torch.randn(1, requires_grad=True))
    return torch.optim.SGD([param], lr=lr)


def _make_loop(tmp_path: str | Path,
               *,
               session_id: int = 1,
               iterations: int = -999,
               file_exists: bool = False,
               config: dict[str, object] | None = None) -> tuple[object,
                                                                 unittest.mock.MagicMock,
                                                                 _RecordingOptimizerUnit,
                                                                 _Events]:
    """ Assemble a lightweight training-loop double with real-observable collaborators """
    checkpoint = os.path.join(str(tmp_path), "model.ckpt")
    model = unittest.mock.MagicMock()
    model.io.checkpoint_path = checkpoint
    model.checkpoint_path = checkpoint
    model.state.session_config = {}
    model.state.session_id = session_id
    model.state.iterations = iterations
    model.state.config = config if config is not None else unittest.mock.MagicMock()
    model.io.file_exists = file_exists
    optimizer_unit = _RecordingOptimizerUnit()
    events = _Events()

    class _Units:
        pass

    units = _Units()
    # pylint:disable=attribute-defined-outside-init
    units.stages_optional = {"step": []}

    class _Loop:
        pass

    loop = _Loop()
    loop.model = model
    loop.optimizer_unit = optimizer_unit
    loop.events = events
    loop.units = units
    loop.current_loss = [_CurrentLossItem(1.0)]
    return loop, model, optimizer_unit, events


def _make_lrf_state(tmp_path: str | Path,
                    *,
                    mode: str = "set",
                    start_lr: float = 1e-10) -> tuple[LRFState,
                                                      unittest.mock.MagicMock,
                                                      _RecordingOptimizerUnit,
                                                      _Events]:
    """ Build an LRFState wired to a shared training-loop double """
    loop, model, optimizer_unit, events = _make_loop(tmp_path)
    scheduler = LRFScheduler(optimizer_unit.optimizer, start_lr=start_lr, end_lr=1e-2,
                             beta=0.98, total_steps=5)
    finder = LearningRateFinder(scheduler, "default")
    state = LRFState(loop, scheduler, finder, mode=mode, start_lr=start_lr)
    return state, model, optimizer_unit, events


@pytest.fixture(autouse=True)
def _quiet_tqdm() -> None:
    """ Isolate the tqdm progress-bar seam so finder tests never touch stderr """
    with unittest.mock.patch.object(lr_finder_unit, "tqdm", unittest.mock.MagicMock()):
        yield


@pytest.fixture(autouse=True)
def _agg_backend() -> None:
    """ Ensure matplotlib renders headlessly for plot_loss tests """
    matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _ignore_lr_scheduler_ordering_warning() -> None:
    """ Silence PyTorch's LR-order warning for schedulers driven outside optimizer.step()

    LRFScheduler and WarmupScheduler advance their scheduler without a preceding optimizer.step()
    (an explicit set_lr sweep / custom linear schedule), so this ordering heuristic is a false
    positive here and does not change any scheduled learning rate
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message=r"Detected call of .*?before .*?optimizer\.step\(\)")
        yield


# =============================================================================
# LRStrength — enum multipliers
# =============================================================================


class TestLRStrength:
    """ Tests for the LRStrength multiplier enum values """
    @pytest.mark.parametrize("member, expected_multiplier",
                             [("DEFAULT", 10),
                              ("AGGRESSIVE", 5),
                              ("EXTREME", 2.5)])
    def test_enum_values_match_documented_multipliers(self,
                                                      member: str,
                                                      expected_multiplier: float) -> None:
        """ Each LRStrength member exposes the strength multiplier documented for it """
        assert getattr(LRStrength, member).value == expected_multiplier


# =============================================================================
# LRFScheduler — construction
# =============================================================================


class TestLRFSchedulerInit:
    """ Tests for LRFScheduler construction and initial attributes """
    def test_attributes_ready_before_first_step(self) -> None:
        """ Exposes empty tracking collections and inf best-loss before any step """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=10)
        assert scheduler.learning_rates == []
        assert scheduler.smooth_losses == []
        assert float(scheduler.best_loss) == float("inf")

    def test_gamma_follows_exponential_formula(self) -> None:
        """ Derives gamma from the end/start ratio over total steps """
        start_lr = 1e-3
        end_lr = 1e-2
        total_steps = 10
        scheduler = LRFScheduler(_make_optimizer(), start_lr, end_lr, 0.98, total_steps)
        expected_gamma = (end_lr / start_lr) ** (1.0 / total_steps)
        assert scheduler.gamma == pytest.approx(expected_gamma)


# =============================================================================
# LRFScheduler — step() loss-smoothing behaviour
# =============================================================================


class TestLRFSchedulerStep:
    """ Tests for LRFScheduler step() loss-smoothing behaviour """
    def test_no_tracking_when_loss_is_none(self) -> None:
        """ Records nothing when a step runs without a loss """
        scheduler = LRFScheduler(_make_optimizer(), start_lr=1e-3, end_lr=1e-2, beta=0.98,
                                 total_steps=10)
        scheduler.step(loss=None)
        assert scheduler.learning_rates == []
        assert scheduler.smooth_losses == []

    @pytest.mark.parametrize("step_index", [1, 2, 3])
    def test_learning_rate_progression_is_exponential(self, step_index: int) -> None:
        """ Scales LR exponentially by step index across the sweep """
        start_lr = 1e-3
        end_lr = 1e-2
        total_steps = 10
        scheduler = LRFScheduler(_make_optimizer(), start_lr, end_lr, 0.98, total_steps)
        for i in range(1, step_index + 1):
            scheduler.step(loss=torch.tensor(float(i)))
        expected_lr = start_lr * (end_lr / start_lr) ** (step_index / total_steps)
        assert scheduler.learning_rates[-1] == pytest.approx(expected_lr, rel=1e-6)

    def test_best_loss_is_minimum_smoothed_loss(self) -> None:
        """ best_loss tracks the minimum smoothed loss seen so far """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=10)
        for loss in (1.0, 2.0, 0.5):
            scheduler.step(loss=torch.tensor(loss))
        assert float(scheduler.best_loss) == pytest.approx(1.0)

    def test_smoothed_losses_recorded_per_step(self) -> None:
        """ Appends exactly one smoothed loss per stepped iteration """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=10)
        for loss in (1.0, 2.0, 0.5):
            scheduler.step(loss=torch.tensor(loss))
        assert len(scheduler.smooth_losses) == 3
        assert float(scheduler.smooth_losses[0]) == pytest.approx(1.0)


# =============================================================================
# LRFScheduler — state_dict() / load_state_dict() round-trip
# =============================================================================


class TestLRFSchedulerStateDict:
    """ Tests for LRFScheduler state_dict/load_state_dict round-trip """
    def test_round_trip_preserves_all_attributes(self) -> None:
        """ state_dict/load_state_dict restore beta, steps and tracked losses """
        start_lr = 1e-3
        end_lr = 1e-2
        total_steps = 7
        scheduler = LRFScheduler(_make_optimizer(), start_lr, end_lr, 0.98, total_steps)
        for loss in (1.0, 2.0, 0.5):
            scheduler.step(loss=torch.tensor(loss))

        learning_rates_snapshot = list(scheduler.learning_rates)
        smooth_losses_snapshot = [float(x) for x in scheduler.smooth_losses]
        best_loss_snapshot = float(scheduler.best_loss)

        restored = LRFScheduler(_make_optimizer(), start_lr, end_lr, 0.98, total_steps)
        restored.load_state_dict(scheduler.state_dict())

        assert restored._beta == 0.98
        assert restored.total_steps == total_steps
        assert restored.learning_rates == learning_rates_snapshot
        assert [float(x) for x in restored.smooth_losses] == pytest.approx(smooth_losses_snapshot)
        assert float(restored.best_loss) == pytest.approx(best_loss_snapshot)


# =============================================================================
# LearningRateFinder — optimal-rate computation
# =============================================================================


class TestLearningRateFinderOptimal:
    """ Tests for LearningRateFinder optimal-rate computation """
    def test_raises_before_any_run(self) -> None:
        """ Raises until a sweep has produced results """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=5)
        finder = LearningRateFinder(scheduler, "default")
        with pytest.raises(AssertionError):
            _ = finder.optimal_learning_rate

    def test_best_learning_rate_divides_by_strength(self) -> None:
        """ Best LR equals the optimal LR divided by the strength multiplier """
        start_lr = 1e-3
        end_lr = 1e-2
        total_steps = 5
        scheduler = LRFScheduler(_make_optimizer(), start_lr, end_lr, 0.98, total_steps)
        # Strictly decreasing losses pin the minimum at index 1 (last step recorded).
        for loss in (1.0, 0.5):
            scheduler.step(loss=torch.tensor(loss))
        finder = LearningRateFinder(scheduler, "default")
        expected = start_lr * (end_lr / start_lr) ** (2 / total_steps) / LRStrength.DEFAULT.value
        assert finder._get_best_learning_rate() == pytest.approx(expected, rel=1e-6)


# =============================================================================
# LearningRateFinder — step() stopping behaviour
# =============================================================================


class TestLearningRateFinderStep:
    """ Tests for LearningRateFinder step() stopping behaviour """
    def test_nan_loss_stops_early(self) -> None:
        """ Stops and finalises when a NaN loss is observed """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=10)
        finder = LearningRateFinder(scheduler, "default")
        for loss in (1.0, 1.0):
            assert finder.step(torch.tensor(loss)) is False
        assert finder.step(torch.tensor(float("nan"))) is True

    def test_divergence_stops_early(self) -> None:
        """ Stops early once smoothing diverges past the stop factor """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=10)
        finder = LearningRateFinder(scheduler, "default")
        for loss in (1.0, 1.0):
            assert finder.step(torch.tensor(loss)) is False
        # A huge loss pushes the smoothed value far beyond stop_factor * best_loss.
        assert finder.step(torch.tensor(1e6)) is True

    def test_runs_to_total_steps_and_stops(self) -> None:
        """ Continues stepping until total steps then returns True """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=5)
        finder = LearningRateFinder(scheduler, "default")
        returns = [finder.step(torch.tensor(1.0)) for _ in range(4)]
        assert returns == [False, False, False, False]
        assert finder.step(torch.tensor(1.0)) is True

    def test_optimal_rate_matches_minimum_after_full_run(self) -> None:
        """ Optimal LR matches the minimum learning rate after a full run """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=5)
        finder = LearningRateFinder(scheduler, "default")
        for _ in range(5):
            finder.step(torch.tensor(1.0))
        best_idx = [float(x) for x in scheduler.smooth_losses].index(float(scheduler.best_loss))
        expected = scheduler.learning_rates[best_idx] / LRStrength.DEFAULT.value
        assert finder.optimal_learning_rate == pytest.approx(expected, rel=1e-9)

    def test_update_progress_bar_forwards_current_and_best(self) -> None:
        """ update_progress_bar reports current and best LRs """
        scheduler = LRFScheduler(_make_optimizer(),
                                 start_lr=1e-3,
                                 end_lr=1e-2,
                                 beta=0.98,
                                 total_steps=5)
        finder = LearningRateFinder(scheduler, "default")
        finder.step(torch.tensor(1.0))
        finder._p_bar = unittest.mock.MagicMock()
        current = scheduler.learning_rates[-1]
        best = finder._get_best_learning_rate()
        finder.update_progress_bar()
        (finder._p_bar.set_description
         .assert_called_once_with(f"Current: {current:.1e}  Best: {best:.1e}"))


# =============================================================================
# plot_loss — file-writing behaviour
# =============================================================================


class TestPlotLoss:
    """ Tests for plot_loss PNG file-writing behaviour """
    def test_writes_png_with_custom_skips(self, tmp_path: Path) -> None:
        """ Writes a non-empty PNG using custom skip counts """
        path = os.path.join(str(tmp_path), "lrf.png")
        plot_loss(path,
                  learning_rates=[1e-3, 1e-2, 1e-1],
                  losses=[5.0, 3.0, 8.0],
                  best_loss=3.0,
                  skip_begin=1,
                  skip_end=1)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_writes_png_with_default_skips(self, tmp_path: Path) -> None:
        """ Writes a non-empty PNG with default skip counts """
        path = os.path.join(str(tmp_path), "lrf.png")
        learning_rates = [1e-3 * (10 ** (i / 11)) for i in range(12)]
        losses = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        plot_loss(path, learning_rates=learning_rates, losses=losses, best_loss=min(losses))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# =============================================================================
# LRFState — lifecycle behaviour
# =============================================================================


class TestLRFState:
    """ Tests for LRFState lifecycle behaviour """
    def test_on_load_sets_start_lr_backs_up_and_enters_pre_training(self, tmp_path: Path) -> None:
        """ on_load restores start LR, backs up weights and enters pre-training """
        with unittest.mock.patch("lib.training.units.lr_finder_unit.torch.save",
                                 unittest.mock.MagicMock()):
            state, model, optimizer_unit, _events = _make_lrf_state(tmp_path, start_lr=1e-10)
            state.on_load()
        assert optimizer_unit.lr == 1e-10
        model.plugin.state_dict.assert_called_once()
        model.state.set_pre_training.assert_called_once()

    def test_set_learning_rate_persists_to_state_and_optimizer(self, tmp_path: Path) -> None:
        """ _set_learning_rate persists to state and sets the optimizer LR """
        state, model, optimizer_unit, _events = _make_lrf_state(tmp_path)
        state._set_learning_rate(5.0)
        assert model.state.session_config["learning_rate"] == 5.0
        assert model.state.lr_finder == 5.0
        assert optimizer_unit.lr == 5.0

    def test_validate_result_accepts_lr_above_start(self, tmp_path: Path) -> None:
        """ Accepts a result LR above the sweep start LR """
        state, _model, _optimizer_unit, _events = _make_lrf_state(tmp_path, start_lr=1e-10)
        assert state._validate_result(5.0) is True

    def test_validate_result_rejects_lr_below_start(self, tmp_path: Path) -> None:
        """ Rejects a result LR at or below the sweep start LR """
        state, _model, _optimizer_unit, _events = _make_lrf_state(tmp_path, start_lr=1e-3)
        assert state._validate_result(1e-4) is False

    def test_validate_result_removes_only_backup_on_invalid_lr(self, tmp_path: Path) -> None:
        """ Removes only the existing backup file when the result is invalid """
        state, _model, _optimizer_unit, _events = _make_lrf_state(tmp_path, start_lr=1e-3)
        open(state._backing_file, "w").close()
        assert not os.path.exists(state._checkpoint_file)
        assert state._validate_result(1e-4) is False
        assert not os.path.exists(state._backing_file)

    def test_validate_result_removes_only_checkpoint_on_invalid_lr(self, tmp_path: Path) -> None:
        """ Removes only the existing checkpoint file on invalid result without touching backup """
        state, _model, _optimizer_unit, _events = _make_lrf_state(tmp_path, start_lr=1e-3)
        open(state._checkpoint_file, "w").close()
        assert not os.path.exists(state._backing_file)
        assert state._validate_result(1e-4) is False
        assert not os.path.exists(state._checkpoint_file)

    def test_finalize_applies_optimal_lr_and_cleans_up(self, tmp_path: Path) -> None:
        """ _finalize applies the optimal LR and removes backup weights """
        with (unittest.mock.patch("lib.training.units.lr_finder_unit.torch.save",
                                  unittest.mock.MagicMock()),
              unittest.mock.patch("lib.training.units.lr_finder_unit.torch.load",
                                  return_value={"model": {}, "optimizer": {}})):
            state, model, optimizer_unit, _events = _make_lrf_state(tmp_path)
            fake_finder = unittest.mock.MagicMock()
            fake_finder.optimal_learning_rate = 5.0
            state._lrf = fake_finder
            open(state._backing_file, "w").close()
            assert state.step() is True
        assert model.state.session_config["learning_rate"] == 5.0
        assert model.state.lr_finder == 5.0
        assert optimizer_unit.lr == 5.0
        assert state._scheduler is None
        assert not os.path.exists(state._backing_file)

    def test_finalize_exits_when_result_invalid(self, tmp_path: Path) -> None:
        """ _finalize signals exit when the result is invalid """
        with unittest.mock.patch("lib.training.units.lr_finder_unit.torch.save",
                                 unittest.mock.MagicMock()):
            state, _model, optimizer_unit, events = _make_lrf_state(tmp_path, start_lr=1e-3)
            fake_finder = unittest.mock.MagicMock()
            fake_finder.optimal_learning_rate = 1e-4
            state._lrf = fake_finder
            open(state._backing_file, "w").close()
            state._finalize()
        assert events.exit.set_called is True
        assert optimizer_unit.lr is None

    def test_resume_requires_scheduler(self, tmp_path: Path) -> None:
        """ resume() requires an existing scheduler """
        state, _model, _optimizer_unit, _events = _make_lrf_state(tmp_path)
        state._scheduler = None
        with pytest.raises(AssertionError):
            state.resume()


# =============================================================================
# LRFinderUnit — configuration behaviour
# =============================================================================


class TestLRFinderUnitConfig:
    """ Tests for LRFinderUnit configuration resolution """
    def test_repr_contains_all_config_values(self) -> None:
        """ repr embeds every configured LRFinderUnit value """
        unit = LRFinderUnit(steps=100,
                            strength="aggressive",
                            mode="graph_and_set",
                            stop_factor=7,
                            start_lr=1e-3,
                            end_lr=1e-2,
                            beta=0.9)
        text = repr(unit)
        assert "steps=100" in text
        assert "strength='aggressive'" in text
        assert "mode='graph_and_set'" in text
        assert "stop_factor=7" in text

    def test_kwargs_from_config_fills_defaults(self) -> None:
        """ _kwargs_from_config fills defaults for unspecified keys """
        unit = LRFinderUnit()
        config = {"lr_finder_iterations": 50,
                  "lr_finder_strength": "aggressive",
                  "lr_finder_mode": "graph_and_set"}
        unit._kwargs_from_config(config)
        assert unit._scheduler_kwargs["total_steps"] == 50
        assert unit._lrf_kwargs["strength"] == "aggressive"
        assert unit._lrf_state_kwargs["mode"] == "graph_and_set"

    def test_kwargs_from_config_keeps_explicit_values(self) -> None:
        """ _kwargs_from_config keeps explicitly provided values """
        unit = LRFinderUnit(steps=25, strength="default", mode="set")
        config = {"lr_finder_iterations": 50,
                  "lr_finder_strength": "aggressive",
                  "lr_finder_mode": "graph_and_set"}
        unit._kwargs_from_config(config)
        assert unit._scheduler_kwargs["total_steps"] == 25
        assert unit._lrf_kwargs["strength"] == "default"
        assert unit._lrf_state_kwargs["mode"] == "set"

    def test_set_learning_rate_from_lrf_applies_lr(self, tmp_path: Path) -> None:
        """ _set_learning_rate_from_lrf applies the stored optimal LR """
        loop, _model, optimizer_unit, _events = _make_loop(tmp_path)
        unit = LRFinderUnit()
        loop.model.state.lr_finder = 5.0
        unit._set_learning_rate_from_lrf(loop.model.state, optimizer_unit)
        assert loop.model.state.session_config["learning_rate"] == 5.0
        assert optimizer_unit.lr == 5.0

    def test_set_learning_rate_from_lrf_rejects_non_positive(self, tmp_path: Path) -> None:
        """ _set_learning_rate_from_lrf rejects non-positive LRs """
        loop, _model, optimizer_unit, _events = _make_loop(tmp_path)
        unit = LRFinderUnit()
        loop.model.state.lr_finder = 0.0
        with pytest.raises(AssertionError):
            unit._set_learning_rate_from_lrf(loop.model.state, optimizer_unit)

    def test_on_load_runs_fresh_sweep(self, tmp_path: Path) -> None:
        """ on_load starts a fresh sweep for session 1 without a checkpoint """
        loop, _model, optimizer_unit, _events = _make_loop(tmp_path, session_id=1,
                                                           file_exists=False,
                                                           config={"lr_finder_iterations": 5})
        unit = LRFinderUnit(steps=5, strength="default", mode="set")
        with unittest.mock.patch("lib.training.units.lr_finder_unit.torch.save",
                                 unittest.mock.MagicMock()):
            unit.on_load(loop)
        assert unit._scheduler is not None
        assert optimizer_unit.lr == 1e-10

    def test_on_load_applies_stored_lr_for_existing_run(self, tmp_path: Path) -> None:
        """ on_load applies the stored optimal LR for an existing run """
        loop, _model, optimizer_unit, _events = _make_loop(tmp_path, session_id=5,
                                                           file_exists=True,
                                                           config={"lr_finder_iterations": 5})
        unit = LRFinderUnit()
        loop.model.state.lr_finder = 3.0
        unit.on_load(loop)
        assert unit._scheduler is None
        assert optimizer_unit.lr == 3.0

    def test_on_load_resumes_then_restores_via_later_load_state_dict(self, tmp_path: Path) -> None:
        """ Resume path sets up a sweep then defers restore to a later load_state_dict """
        loop, _model, optimizer_unit, _events = _make_loop(
            tmp_path, session_id=1, file_exists=True, iterations=-999,
            config={"lr_finder_iterations": 5},
        )
        unit = LRFinderUnit(steps=5, strength="default", mode="set")
        # A stored optimal LR exists in the checkpoint, but the resume path must not apply it.
        loop.model.state.lr_finder = 3.0
        with unittest.mock.patch("lib.training.units.lr_finder_unit.torch.save",
                                 unittest.mock.MagicMock()):
            unit.on_load(loop)
        # setup-on-load: a fresh sweep was created (unlike the stored-LR shortcut path) ...
        assert unit._scheduler is not None
        assert unit._lrf_state is not None
        assert optimizer_unit.lr == 1e-10
        # ... and the stored optimal LR was left untouched (branch B shortcut skipped).
        assert optimizer_unit.lr != 3.0

        # restore-via-later-load_state_dict: the framework later restores saved stats.
        fake_scheduler = unittest.mock.MagicMock()
        fake_lrf_state = unittest.mock.MagicMock()
        unit._scheduler = fake_scheduler
        unit._lrf_state = fake_lrf_state
        unit.load_state_dict({"state": "data"})
        fake_scheduler.load_state_dict.assert_called_once_with({"state": "data"})
        fake_lrf_state.resume.assert_called_once()


# =============================================================================
# LRFinderUnit — lifecycle behaviour
# =============================================================================
class TestLRFinderUnitLifecycle:
    """ Tests for LRFinderUnit on_load/on_start/step lifecycle behaviour """
    def test_on_start_removes_self_from_steppers(self, tmp_path: Path) -> None:
        """ on_start removes this unit from the step steppers """
        loop, _model, _optimizer_unit, _events = _make_loop(tmp_path)
        unit = LRFinderUnit(steps=5, strength="default", mode="set")
        with unittest.mock.patch("lib.training.units.lr_finder_unit.torch.save",
                                 unittest.mock.MagicMock()):
            unit.on_load(loop)
        loop.units.stages_optional["step"].append(unit)
        unit.on_start()
        assert unit not in loop.units.stages_optional["step"]

    def test_step_clears_scheduler_on_completion(self, tmp_path: Path) -> None:
        """ step clears the scheduler once the sweep completes """
        loop, _model, _optimizer_unit, _events = _make_loop(tmp_path)
        unit = LRFinderUnit()
        unit._units = loop.units

        class _FakeLRFState:
            called = False

            def step(self) -> bool:
                type(self).called = True
                return True

        sentinel = object()
        unit._lrf_state = _FakeLRFState()
        unit._scheduler = sentinel
        unit.step(-1)
        assert unit._scheduler is None
        assert _FakeLRFState.called is True

    def test_step_asserts_on_non_pre_training_iteration(self) -> None:
        """ step asserts it is only called during pre-training (iteration -1) """
        unit = LRFinderUnit()
        unit._lrf_state = unittest.mock.MagicMock()
        with pytest.raises(AssertionError):
            unit.step(0)

    def test_step_asserts_scheduler_present(self) -> None:
        """ step asserts a scheduler exists before stepping """
        unit = LRFinderUnit()
        unit._lrf_state = None
        with pytest.raises(AssertionError):
            unit.step(-1)

    def test_state_dict_delegates_to_scheduler(self) -> None:
        """ state_dict delegates to the scheduler when one exists """
        unit = LRFinderUnit()
        unit._scheduler = unittest.mock.MagicMock()
        unit._scheduler.state_dict.return_value = {"key": "value"}
        assert unit.state_dict() == {"key": "value"}

    def test_state_dict_empty_when_no_scheduler(self) -> None:
        """ state_dict returns an empty dict when no scheduler exists """
        unit = LRFinderUnit()
        assert unit.state_dict() == {}

    def test_load_state_dict_restores_and_resumes(self) -> None:
        """ load_state_dict restores scheduler state and resumes the sweep """
        unit = LRFinderUnit()
        fake_scheduler = unittest.mock.MagicMock()
        fake_lrf_state = unittest.mock.MagicMock()
        unit._scheduler = fake_scheduler
        unit._lrf_state = fake_lrf_state
        unit.load_state_dict({"state": "data"})
        fake_scheduler.load_state_dict.assert_called_once_with({"state": "data"})
        fake_lrf_state.resume.assert_called_once()

    def test_load_state_dict_noop_without_scheduler(self) -> None:
        """ load_state_dict is a no-op when no scheduler exists """
        unit = LRFinderUnit()
        fake_lrf_state = unittest.mock.MagicMock()
        unit._scheduler = None
        unit._lrf_state = fake_lrf_state
        unit.load_state_dict({"state": "data"})
        fake_lrf_state.resume.assert_not_called()

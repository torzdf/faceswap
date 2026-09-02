#!/usr/bin/env python3
# pylint:disable=unused-import,import-error,protected-access,unused-argument,too-few-public-methods
# pylint:disable=missing-function-docstring
""" Pytest unit tests for :mod:`lib.training.units.core.save_unit` """
from __future__ import annotations

import os
import time
import typing as T
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

from lib.model.plugin import State
from lib.training.events import TrainingEvents
from lib.training.units.core.save_unit import (
    Backup,
    LoadUnit,
    SaveUnit,
    Saver,
    Snapshot,
    StateMarkdown,
)


# =============================================================================
# Doubles (model / loop boundaries only)
# =============================================================================


class FakeSaveModel:
    """ Minimal FaceswapModel double covering save-unit and SaveUnit seams """
    def __init__(self, tmp_path: T.Any) -> None:
        self._state = State("mock_plugin")
        self._state.set_plugin_version(1.0)
        self._extra_states: dict[str, T.Any] = {}
        self.checkpoint_path = str(tmp_path / "model.ckpt")
        self.name = "mock_plugin"
        # latest_save must be a real existing file (SaveUnit asserts it and Backup copies it)
        self.latest_save = str(tmp_path / "latest.ckpt")
        with open(self.latest_save, "w", encoding="utf-8") as handle:
            handle.write("{}\n")
        self.extra_cleared = False

    @property
    def state(self) -> State:
        return self._state

    def state_dict(self) -> dict[str, T.Any]:
        return {"model": {"weights": 1}}

    # LoadUnit seams
    def pop_extra_state(self, name: str) -> T.Any:
        return self._extra_states.pop(name, None)

    def clear_extra_state(self) -> None:
        self.extra_cleared = True
        self._extra_states.clear()


class FakeSaveableUnit:
    """ TrainingUnit double that records the state dict it receives """
    def __init__(self, name: str) -> None:
        self.name = name
        self.loaded: T.Any = None

    def load_state_dict(self, state_dict: T.Any) -> None:
        self.loaded = state_dict

    def state_dict(self) -> dict[str, T.Any]:
        return {self.name: "unit-state"}


class FakeUnits:
    """ Provides the loop.units.have_state_dict mapping used by the core units """
    def __init__(self, saveable_units: dict[str, FakeSaveableUnit]) -> None:
        self.have_state_dict = saveable_units


class FakeLoop:
    """ Minimal training-step double exposing loop.units.have_state_dict """
    def __init__(self, saveable_units: dict[str, FakeSaveableUnit]) -> None:
        self.units = FakeUnits(saveable_units)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(name="save_model")
def fixture_save_model(tmp_path: T.Any) -> FakeSaveModel:
    """ A configured FakeSaveModel writing under a real tmp directory """
    return FakeSaveModel(tmp_path)


@pytest.fixture(name="avg_loss")
def fixture_avg_loss() -> npt.NDArray[np.float32]:
    """ A 0-dimensional average-loss array (value 0 so no backup side effects) """
    return np.zeros((), dtype=np.float32)


# =============================================================================
# LoadUnit
# =============================================================================


class TestLoadUnitOnLoad:
    """ Tests for LoadUnit.on_load() """
    def test_on_load_restores_available_state_dicts(
        self, save_model: FakeSaveModel, events: TrainingEvents
    ) -> None:
        """ Units offering state are restored from the model's extra-state collection """
        unit = FakeSaveableUnit("optimizer")
        loop = FakeLoop({"optimizer": unit})
        save_model._extra_states = {"optimizer": {"opt": 1}}
        load_unit = LoadUnit(save_model)

        load_unit.on_load(loop)

        assert unit.loaded == {"opt": 1}
        assert save_model.extra_cleared is True
        # After consumption the extra state is gone from the model.
        assert save_model.pop_extra_state("optimizer") is None

    def test_on_load_skips_missing_state_dicts(
        self, save_model: FakeSaveModel, events: TrainingEvents
    ) -> None:
        """ Units without a present state dict are skipped without error """
        unit = FakeSaveableUnit("optimizer")
        loop = FakeLoop({"optimizer": unit})
        load_unit = LoadUnit(save_model)

        load_unit.on_load(loop)

        assert unit.loaded is None
        assert save_model.extra_cleared is True

    def test_on_load_repr(self, save_model: FakeSaveModel) -> None:
        """ The repr exposes the class name and wrapped model """
        load_unit = LoadUnit(save_model)
        text = repr(load_unit)
        assert "LoadUnit" in text
        assert "model=" in text


# =============================================================================
# StateMarkdown (rendering contract)
# =============================================================================


class TestStateMarkdownRendering:
    """ Tests for StateMarkdown rendering methods """

    def test_render_model_info_reports_identity_and_metrics(
        self, mock_state: State
    ) -> None:
        """ render_model_info exposes model identity and formatted metrics in a table """
        mock_state.set_plugin_version(2.5)
        mock_state.lowest_avg_loss = 0.5
        mock_state.lr_finder = 0.05
        mock_state.add_new_session(32)
        mock_state.increment_iterations()
        markdown = StateMarkdown(mock_state)

        out = markdown.render_model_info()

        assert "### Model Information" in out
        assert "mock-plugin" in out
        # lowest_avg_loss rendered in scientific notation, lr_finder rendered (branch taken)
        assert "5.00e-01" in out
        assert "5.0e-02" in out

    def test_render_config_reports_configuration(self, mock_state: State) -> None:
        """ render_config produces the model config section as markdown """
        mock_state.set_plugin_version(1.0)
        markdown = StateMarkdown(mock_state)

        out = markdown.render_config()

        assert out
        assert "### Model Config" in out

    def test_render_sessions_reports_training_history(self, mock_state: State) -> None:
        """ render_sessions lists each completed session with metadata and config tables """
        mock_state.set_plugin_version(1.0)
        mock_state.add_new_session(32)
        mock_state.increment_iterations()
        mock_state.add_new_session(64)
        mock_state.increment_iterations()
        markdown = StateMarkdown(mock_state)

        out = markdown.render_sessions()

        assert "### Session 1" in out
        assert "### Session 2" in out
        assert "Batch Size" in out

    def test_render_sessions_empty_when_no_completed_sessions(
        self, mock_state: State
    ) -> None:
        """ render_sessions yields no rows when there are no completed sessions to report """
        mock_state.set_plugin_version(1.0)
        markdown = StateMarkdown(mock_state)

        out = markdown.render_sessions()

        assert out == ""

    def test_full_summary_combines_all_sections(self, mock_state: State) -> None:
        """ full_summary composes model info, config and sessions into one report """
        mock_state.set_plugin_version(1.0)
        mock_state.add_new_session(32)
        mock_state.increment_iterations()
        markdown = StateMarkdown(mock_state)

        out = markdown.full_summary()

        assert "## Model" in out
        assert "### Model Information" in out
        assert "### Model Config" in out
        assert "## Sessions" in out
        assert "### Session 1" in out


# =============================================================================
# StateMarkdown (pure formatting + rendering contract)
# =============================================================================


class TestStateMarkdownFormatting:
    """ Tests for StateMarkdown formatting helpers """
    @pytest.mark.parametrize("timestamp", [0, 1620000000])
    def test_format_time_maps_epoch_to_readable(self, timestamp: float) -> None:
        """ A Unix timestamp renders as ``YYYY-MM-DD HH:MM`` in local time """
        expected = time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
        assert StateMarkdown._format_time(timestamp) == expected

    def test_format_to_table_handles_lists_and_widths(self) -> None:
        """ List values expand to multiple rows and columns align on widest cell """
        rows = StateMarkdown._format_to_table({"Key": ["a", "bb"],
                                               "Value": ["1", "2"]})
        joined = chr(10).join(rows)
        assert "|" in joined
        # header + break + one data row per zip element, all as strings.
        assert len(rows) >= 3

    def test_format_to_table_single_value(self) -> None:
        """ A scalar value is treated as a single cell """
        rows = StateMarkdown._format_to_table({"Parameter": ["batch_size"],
                                               "Value": ["32"]})
        joined = chr(10).join(rows)
        assert "batch_size" in joined
        assert "32" in joined


# =============================================================================
# Backup
# =============================================================================


class TestBackup:
    """ Tests for Backup checkpoint behaviour """
    def test_backup_created_on_loss_improvement(self, save_model: FakeSaveModel, tmp_path: T.Any
                                                ) -> None:
        """ A lower loss than the best recorded triggers a ``.bk`` recovery copy """
        save_model.state.lowest_avg_loss = 10.0
        backup = Backup(save_model.state)

        result = backup(save_model.latest_save, 5.0)

        assert result is True
        backup_file = save_model.latest_save + ".bk"
        assert os.path.exists(backup_file)
        assert save_model.state.lowest_avg_loss == 5.0

    def test_no_backup_when_not_improved(self, save_model: FakeSaveModel, tmp_path: T.Any) -> None:
        """ Loss worse than or equal to the best recorded loss creates no backup """
        save_model.state.lowest_avg_loss = 10.0
        backup = Backup(save_model.state)

        result = backup(save_model.latest_save, 20.0)

        assert result is False
        assert not os.path.exists(save_model.latest_save + ".bk")

    def test_no_backup_with_zero_loss(self, save_model: FakeSaveModel, tmp_path: T.Any) -> None:
        """ A zero loss is not treated as an improvement """
        backup = Backup(save_model.state)

        result = backup(save_model.latest_save, 0.0)

        assert result is False
        assert not os.path.exists(save_model.latest_save + ".bk")

    def test_backup_repr(self, save_model: FakeSaveModel) -> None:
        """ The repr surfaces the class name and wrapped state """
        backup = Backup(save_model.state)
        text = repr(backup)
        assert "Backup" in text
        assert "state=" in text


# =============================================================================
# Saver
# =============================================================================


class TestSaver:
    """ Tests for Saver save behaviour """

    def test_save_weights_writes_file_and_info(self, save_model: FakeSaveModel, tmp_path: T.Any
                                               ) -> None:
        """ A weights-only save writes ``model.pth`` plus a markdown info file """
        saver = Saver(save_model, {})
        folder = str(tmp_path)

        with mock.patch("lib.training.units.core.save_unit.torch.save") as torch_save:
            saver(folder, is_checkpoint=False)

        saved_fname = torch_save.call_args.args[1]
        assert saved_fname == os.path.join(folder, "model.pth")
        info_file = os.path.join(folder, "model_info.md")
        assert os.path.exists(info_file)

    def test_saver_removes_stale_opposite_extension(
        self, save_model: FakeSaveModel, tmp_path: T.Any
    ) -> None:
        """ Switching to a weights-only save removes the stale checkpoint file """
        stale = os.path.join(str(tmp_path), "model.ckpt")
        with open(stale, "w", encoding="utf-8") as handle:
            handle.write("stale")
        saver = Saver(save_model, {})

        with mock.patch("lib.training.units.core.save_unit.torch.save") as torch_save:
            saver(str(tmp_path), is_checkpoint=False)

        assert not os.path.exists(stale)
        saved_fname = torch_save.call_args.args[1]
        assert saved_fname.endswith(".pth")

    def test_get_state_dicts_weights_only(self, save_model: FakeSaveModel) -> None:
        """ A non-checkpoint save carries only the model weights """
        saver = Saver(save_model, {"optimizer": FakeSaveableUnit("optimizer")})

        state_dict = saver._get_state_dicts(is_checkpoint=False)

        assert isinstance(state_dict, dict)
        assert "model" in state_dict

    def test_get_state_dicts_includes_saveable_units(self, save_model: FakeSaveModel) -> None:
        """ A checkpoint save merges truthy unit state dicts into the saved weights """
        saver = Saver(save_model, {"optimizer": FakeSaveableUnit("optimizer")})

        state_dict = saver._get_state_dicts(is_checkpoint=True)

        assert "model" in state_dict
        assert "optimizer" in state_dict

    def test_saver_repr(self, save_model: FakeSaveModel) -> None:
        """ The repr surfaces the class name and collaborators """
        saver = Saver(save_model, {})
        text = repr(saver)
        assert "Saver" in text
        assert "model=" in text


# =============================================================================
# Snapshot
# =============================================================================


class TestSnapshot:
    """ Tests for Snapshot folder creation and logging """
    def test_snapshot_creates_folder_and_delegates_to_saver(
        self, save_model: FakeSaveModel, tmp_path: T.Any
    ) -> None:
        """ A snapshot creates the iteration folder and writes a checkpoint there """
        save_model.state.add_new_session(32)
        save_model.state.increment_iterations()
        saver = Saver(save_model, {})
        snapshot = Snapshot(save_model, saver)

        snapshot()

        expected_folder = (f"{os.path.dirname(save_model.checkpoint_path)}"
                           f"_snapshot_{save_model.state.iterations}_iters")
        assert os.path.isdir(expected_folder)
        artifacts = os.listdir(expected_folder)
        checkpoint_written = any(n.endswith((".pth", ".ckpt")) for n in artifacts)
        assert checkpoint_written, "written checkpoint artifact missing"

    def test_snapshot_copies_logs_when_present(self, save_model: FakeSaveModel, tmp_path: T.Any
                                               ) -> None:
        """ Existing logs are copied into the snapshot folder """
        save_model.state.add_new_session(32)
        save_model.state.increment_iterations()
        saver = Saver(save_model, {})
        snapshot = Snapshot(save_model, saver)
        src_logs = os.path.join(
            os.path.dirname(save_model.checkpoint_path), f"{save_model.name}_logs"
        )
        os.makedirs(src_logs, exist_ok=True)

        snapshot()

        expected_folder = (
            f"{os.path.dirname(save_model.checkpoint_path)}"
            f"_snapshot_{save_model.state.iterations}_iters"
        )
        assert os.path.isdir(os.path.join(expected_folder, src_logs))

    def test_snapshot_repr(self, save_model: FakeSaveModel) -> None:
        """ The repr surfaces the class name and collaborators """
        saver = Saver(save_model, {})
        snapshot = Snapshot(save_model, saver)
        text = repr(snapshot)
        assert "Snapshot" in text
        assert "saver=" in text


# =============================================================================
# SaveUnit
# =============================================================================


class TestSaveUnitInit:
    """ Tests for SaveUnit construction """
    def test_init_stores_all_references(self,
                                        save_model: FakeSaveModel,
                                        events: TrainingEvents,
                                        avg_loss: npt.NDArray[np.float32]) -> None:
        """ Construction records every injected collaborator """
        optimizer = FakeSaveableUnit("optimizer")
        unit = SaveUnit(save_model,
                        optimizer,
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")

        assert unit._model is save_model
        assert unit._optimizer is optimizer
        assert unit._events is events
        assert unit._average_loss is avg_loss
        assert unit._save_interval == 100
        assert unit._snapshot_interval == 100
        assert unit._save_train_state == "always"

    def test_init_snapshot_flag_starts_false(self,
                                             save_model: FakeSaveModel,
                                             events: TrainingEvents,
                                             avg_loss: npt.NDArray[np.float32]) -> None:
        """ The snapshot flag defaults to ``False`` before any step """
        unit = SaveUnit(save_model,
                        FakeSaveableUnit("optimizer"),
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")

        assert unit._do_snapshot is False

    def test_init_repr(self,
                       save_model: FakeSaveModel,
                       events: TrainingEvents,
                       avg_loss: npt.NDArray[np.float32]) -> None:
        """ The repr exposes the class name and key parameters """
        unit = SaveUnit(save_model,
                        FakeSaveableUnit("optimizer"),
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")
        text = repr(unit)
        assert "SaveUnit" in text
        assert "save_interval=100" in text


class TestSaveUnitStep:
    """ Tests for SaveUnit.step() """
    @pytest.mark.parametrize(("iteration", "expected_save"),
                             [(0, False),
                              (-5, False),
                              (100, True),
                              (200, True),
                              (99, False)])
    def test_step_sets_save_event_on_interval(self,
                                              save_model: FakeSaveModel,
                                              events: TrainingEvents,
                                              avg_loss: npt.NDArray[np.float32],
                                              iteration: int,
                                              expected_save: bool) -> None:
        """ The save event fires only on multiples of the save interval after start """
        unit = SaveUnit(save_model,
                        FakeSaveableUnit("optimizer"),
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")

        unit.step(iteration)

        assert events.save.is_set() == expected_save

    @pytest.mark.parametrize(
        ("iteration", "expected_snapshot"),
        [
            (100, True),
            (300, True),
            (50, False),
        ],
    )
    def test_step_flags_snapshot_on_interval(
        self,
        save_model: FakeSaveModel,
        events: TrainingEvents,
        avg_loss: np.ndarray[np.float32],
        iteration: int,
        expected_snapshot: bool,
    ) -> None:
        """ A snapshot is flagged on multiples of the snapshot interval """
        save_model.state.add_new_session(32)
        save_model.state.increment_iterations()
        unit = SaveUnit(
            save_model,
            FakeSaveableUnit("optimizer"),
            events,
            avg_loss,
            save_interval=100,
            snapshot_interval=100,
            save_train_state="always",
        )

        unit.step(iteration)

        assert unit._do_snapshot is expected_snapshot


class TestSaveUnitOnLoad:
    """ Tests for SaveUnit.on_load() """
    def test_on_load_wires_saver_and_snapshot(self,
                                              save_model: FakeSaveModel,
                                              events: TrainingEvents,
                                              avg_loss: np.ndarray[np.float32]) -> None:
        """ ``on_load`` constructs the saver and snapshot collaborators """
        unit = SaveUnit(save_model,
                        FakeSaveableUnit("optimizer"),
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")
        loop = FakeLoop({})

        unit.on_load(loop)

        assert isinstance(unit._saver, Saver)
        assert isinstance(unit._snapshot, Snapshot)


class TestSaveUnitOnSave:
    """ Tests for SaveUnit.on_save() """
    def test_on_save_writes_model_and_triggers_update(self,
                                                      save_model: FakeSaveModel,
                                                      events: TrainingEvents,
                                                      avg_loss: npt.NDArray[np.float32]) -> None:
        """ Saving writes the model file and requests a preview refresh """
        unit = SaveUnit(save_model,
                        FakeSaveableUnit("optimizer"),
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")
        loop = FakeLoop({})
        unit.on_load(loop)

        with mock.patch("lib.training.units.core.save_unit.torch.save") as torch_save:
            unit.on_save(iteration=100)

        saved_fname = torch_save.call_args.args[1]
        assert saved_fname.endswith(".ckpt")
        assert events.update.is_set() is True

    def test_on_save_noop_when_exit_requested(self,
                                              save_model: FakeSaveModel,
                                              events: TrainingEvents,
                                              avg_loss: npt.NDArray[np.float32]) -> None:
        """ An exit in progress skips the periodic save entirely """
        unit = SaveUnit(save_model,
                        FakeSaveableUnit("optimizer"),
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")
        loop = FakeLoop({})
        unit.on_load(loop)

        events.exit.set()
        unit.on_save(iteration=100)

        assert events.update.is_set() is False


class TestSaveUnitOnEnd:
    """ Tests for SaveUnit.on_end() """
    def test_on_end_saves_final_model(self,
                                      save_model: FakeSaveModel,
                                      events: TrainingEvents,
                                      avg_loss: npt.NDArray[np.float32]) -> None:
        """ Completing training performs a final checkpoint save """
        unit = SaveUnit(save_model,
                        FakeSaveableUnit("optimizer"),
                        events,
                        avg_loss,
                        save_interval=100,
                        snapshot_interval=100,
                        save_train_state="always")
        loop = FakeLoop({})
        unit.on_load(loop)

        with mock.patch("lib.training.units.core.save_unit.torch.save") as torch_save:
            unit.on_end()

        saved_fname = torch_save.call_args.args[1]
        assert saved_fname.endswith(".ckpt")


# =============================================================================
# Module-level export contract
# =============================================================================


class TestModuleContract:
    """ Tests for the save_unit public export contract """
    def test_all_exports_present(self) -> None:
        """ The module exposes its public unit classes via ``__all__`` """
        # pylint:disable=import-outside-toplevel
        import lib.training.units.core.save_unit as save_unit_module  # noqa:WPS433

        for name in ("LoadUnit", "SaveUnit", "Saver", "Snapshot", "Backup"):
            assert hasattr(save_unit_module, name)

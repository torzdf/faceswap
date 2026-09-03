#!/usr/bin/env python3
# pylint:disable=protected-access, import-error, too-few-public-methods
""" Pytest unit tests for :mod:`lib.training.units.tensorboard_unit` """
from __future__ import annotations

import os
import struct
import typing as T

import pytest
import torch
from torch import nn

from lib.training.units.tensorboard_unit import TensorBoardUnit, RecordIterator
from tests.lib.training.loss.batch_loss_mock import BatchLossMock

if T.TYPE_CHECKING:
    from pathlib import Path
    from lib.training.loss import BatchLoss


# =============================================================================
# Test doubles and helpers
# =============================================================================


class RecordingWriter:
    """ Records add_scalar/flush/close calls for black-box assertions

    A lightweight stand-in for ``SummaryWriter`` that captures the scalar stream and lifecycle into
    plain attributes, so tests can assert on observable output rather than inspecting a mock's call
    mechanics """
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.graphs: list[tuple[object, object]] = []
        self.flushed: bool = False
        self.closed: bool = False

    def add_scalar(self, tag: str, value: torch.Tensor | float, global_step: int = 0
                   ) -> None:
        """ Record a scalar metric for later assertion """
        numeric = value.item() if isinstance(value, torch.Tensor) else float(value)
        self.scalars.append((tag, numeric, global_step))

    def flush(self) -> None:
        """ Flag that the writer was flushed """
        self.flushed = True

    def close(self) -> None:
        """ Flag that the writer was closed """
        self.closed = True

    def add_graph(self, model: object, inputs: object) -> None:
        """ Record a traced graph for later assertion """
        self.graphs.append((model, inputs))


def _build_unit(writer: RecordingWriter) -> TensorBoardUnit:
    """ Build a TensorBoardUnit without disk writes or a real SummaryWriter """
    unit = TensorBoardUnit.__new__(TensorBoardUnit)
    unit.log_name = "[TensorBoard]"
    unit._writer = writer
    unit._current_loss = []
    return unit


def _record(payload: bytes) -> bytes:
    """ Wrap ``payload`` in a valid TensorBoard event record frame """
    header = struct.pack("Q", len(payload))  # native order matches unpack('Q')
    return header + b"\x00\x00\x00\x00" + payload + b"\x11\x22\x33\x44"


def _single_component() -> BatchLossMock:
    """ One loss component, one batch sample """
    return BatchLossMock(unweighted=[{"l1": torch.tensor(1.0)}],
                         weighted=[{"l1": torch.tensor(1.5)}])


def _multi_sample() -> BatchLossMock:
    """ One loss component across two batch samples (indexed labels) """
    return BatchLossMock(unweighted=[{"l1": torch.tensor(1.0)}, {"l1": torch.tensor(2.0)}],
                         weighted=[{"l1": torch.tensor(1.5)}, {"l1": torch.tensor(3.0)}])


def _masked() -> BatchLossMock:
    """ One loss component with a learn-mask tensor """
    return BatchLossMock(unweighted=[{"l1": torch.tensor(2.0)}],
                         weighted=[{"l1": torch.tensor(2.0)}],
                         mask=torch.tensor(4.0))


class _FakeModelInfo:
    input_shapes: list[tuple[int, int, int]] = [(3, 256, 256)]


class _TracerModule(nn.Module):
    """ Minimal real module so graph tracing exercises a genuine training-mode model """
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(256, 1)

    def forward(self, *args) -> torch.Tensor:  # pylint:disable=missing-function-docstring
        return self.linear(args[0])


class _FakeModel:
    def __init__(self, plugin: object = None) -> None:
        self.plugin: object = plugin if plugin is not None else object()
        self.device = torch.device("cpu")
        self.info = _FakeModelInfo()


class _FakeLoop:
    def __init__(self, current_loss: list["BatchLoss"], plugin: object = None) -> None:
        self.current_loss = current_loss
        self.device = torch.device("cpu")
        self.model = _FakeModel(plugin=plugin)


# =============================================================================
# RecordIterator — reading event files
# =============================================================================


def test_yields_full_record_payload_then_eofs(tmp_path: Path) -> None:
    """ A well-formed file yields its payload verbatim, then stops """
    log_file = tmp_path / "events.log"
    log_file.write_bytes(_record(b"event-data"))

    iterator = RecordIterator(str(log_file))
    assert next(iterator) == b"event-data"
    with pytest.raises(StopIteration):
        next(iterator)


def test_yields_multiple_records_in_order(tmp_path: Path) -> None:
    """ Records are returned in file order until the stream is exhausted """
    log_file = tmp_path / "events.log"
    log_file.write_bytes(_record(b"first") + _record(b"second"))

    iterator = RecordIterator(str(log_file))
    assert next(iterator) == b"first"
    assert next(iterator) == b"second"
    with pytest.raises(StopIteration):
        next(iterator)


def test_raises_stop_iteration_on_empty_file(tmp_path: Path) -> None:
    """ An empty file stops immediately without raising anything else """
    log_file = tmp_path / "events.log"
    log_file.write_bytes(b"")

    with pytest.raises(StopIteration):
        next(RecordIterator(str(log_file)))


def test_raises_stop_iteration_on_partial_header(tmp_path: Path) -> None:
    """ A header shorter than 8 bytes is treated as a partial record """
    log_file = tmp_path / "events.log"
    log_file.write_bytes(b"\x01\x02\x03")

    with pytest.raises(StopIteration):
        next(RecordIterator(str(log_file)))


def test_raises_stop_iteration_on_implausible_length(tmp_path: Path) -> None:
    """ A record length beyond the maximum is rejected as a partial read """
    log_file = tmp_path / "events.log"
    log_file.write_bytes(struct.pack("Q", RecordIterator._max_record_size + 1))

    with pytest.raises(StopIteration):
        next(RecordIterator(str(log_file)))


def test_raises_stop_iteration_on_partial_data(tmp_path: Path) -> None:
    """ Declaring a length longer than the remaining bytes stops gracefully """
    log_file = tmp_path / "events.log"
    body = struct.pack("Q", 100) + b"\x00\x00\x00\x00" + b"abcd"
    log_file.write_bytes(body)

    with pytest.raises(StopIteration):
        next(RecordIterator(str(log_file)))


def test_live_iterator_reopens_and_seeks_to_saved_position(tmp_path: Path) -> None:
    """ A live iterator re-opens a closed file at its saved position after new data is appended """
    log_file = tmp_path / "events.log"
    log_file.write_bytes(b"")  # start empty

    iterator = RecordIterator(str(log_file), is_live=True)

    with pytest.raises(StopIteration):
        next(iterator)  # partial header -> live EOF saves position and closes the file
    assert iterator._log_file.closed is True

    appended = _record(b"live-data")
    log_file.write_bytes(appended)  # append one record at offset 0

    payload = next(iterator)  # reopen + seek to saved position, then read the new record
    assert payload == b"live-data"


# =============================================================================
# TensorBoardUnit — __repr__
# =============================================================================


class TestTensorBoardUnitRepr:
    """ Tests for TensorBoardUnit.__repr__ """
    def test_repr_contains_class_and_config(self) -> None:
        """ repr exposes the class name and key configuration for debugging """
        unit = TensorBoardUnit.__new__(TensorBoardUnit)
        unit._model_folder = "mymodel"
        unit._model_name = "gan"
        unit._session_id = 3
        unit._write_graph = True

        repr_str = repr(unit)
        assert "TensorBoardUnit" in repr_str
        assert "model_folder='mymodel'" in repr_str
        assert "session_id=3" in repr_str


# =============================================================================
# TensorBoardUnit — __init__
# =============================================================================


class TestTensorBoardUnitInit:
    """ Tests for TensorBoardUnit construction and log-dir derivation """
    def test_init_derives_expected_log_dir_and_stores_config(self, monkeypatch: pytest.MonkeyPatch
                                                             ) -> None:
        """ Construction logs under model_folder/name_logs/session_id/train """
        captured: dict[str, str] = {}

        def _fake_writer(log_dir: str) -> None:
            captured["log_dir"] = log_dir

        monkeypatch.setattr("lib.training.units.tensorboard_unit.SummaryWriter", _fake_writer)

        unit = TensorBoardUnit(model_folder="mymodel", model_name="gan", session_id=2)

        assert captured["log_dir"] == os.path.join("mymodel", "gan_logs", "session_2", "train")
        assert unit._session_id == 2
        assert unit._write_graph is True


# =============================================================================
# TensorBoardUnit — _get_logs
# =============================================================================


class TestTensorBoardUnitGetLogs:
    """ Tests for _get_logs metric extraction """
    def test_single_component_maps_to_face_weighted_unweighted_keys(self) -> None:
        """ A single-component loss maps to face/weighted/unweighted 'A' keys """
        unit = _build_unit(RecordingWriter())
        logs = unit._get_logs([_single_component()])

        assert logs["total"].item() == pytest.approx(1.5)
        assert logs["face_A"].item() == pytest.approx(1.5)
        assert logs["weighted_A"]["l1"].item() == pytest.approx(1.5)
        assert logs["unweighted_A"]["l1"].item() == pytest.approx(1.0)

    def test_multi_sample_loss_uses_indexed_labels(self) -> None:
        """ Multiple samples produce per-sample indexed metric keys """
        unit = _build_unit(RecordingWriter())
        logs = unit._get_logs([_multi_sample()])

        assert set(logs) == {"total", "face_A_0", "face_A_1",
                             "weighted_A_0", "unweighted_A_0",
                             "weighted_A_1", "unweighted_A_1"}
        assert logs["face_A_0"].item() == pytest.approx(1.5)
        assert logs["face_A_1"].item() == pytest.approx(3.0)

    def test_mask_loss_is_logged_and_totals_include_mask(self) -> None:
        """ A learn-mask is logged under its label and added to the total """
        unit = _build_unit(RecordingWriter())
        logs = unit._get_logs([_masked()])

        assert logs["mask_A"].item() == pytest.approx(4.0)
        assert logs["total"].item() == pytest.approx(6.0)


# =============================================================================
# TensorBoardUnit — step
# =============================================================================


class TestTensorBoardUnitStep:
    """ Tests for step batch logging """
    def test_skips_pretraining_iteration_without_logging(self) -> None:
        """ Negative (pre-training) iterations log nothing """
        recording = RecordingWriter()
        unit = _build_unit(recording)
        unit._current_loss = [_multi_sample()]

        unit.step(-1)
        assert not recording.scalars

    def test_logs_batch_scalars_at_expected_tags_and_steps(self) -> None:
        """ step logs each extracted metric under a batch_ tag at the iteration """
        recording = RecordingWriter()
        unit = _build_unit(recording)
        unit._current_loss = [_multi_sample()]

        unit.step(10)

        expected = [("batch_total", 4.5, 10),
                    ("batch_face_A_0", 1.5, 10),
                    ("batch_weighted_A_0/l1", 1.5, 10),
                    ("batch_unweighted_A_0/l1", 1.0, 10),
                    ("batch_face_A_1", 3.0, 10),
                    ("batch_weighted_A_1/l1", 3.0, 10),
                    ("batch_unweighted_A_1/l1", 2.0, 10)]
        assert recording.scalars == expected


# =============================================================================
# TensorBoardUnit — lifecycle: on_save / on_end / on_load
# =============================================================================


class TestTensorBoardUnitLifecycle:
    """ Tests for on_save/on_end/on_load lifecycle behaviour """
    def test_on_save_flushes_writer(self) -> None:
        """ on_save flushes the writer at save intervals """
        recording = RecordingWriter()
        unit = _build_unit(recording)

        unit.on_save(500)
        assert recording.flushed is True

    def test_on_end_flushes_and_closes_writer(self) -> None:
        """ on_end flushes and then closes the writer """
        recording = RecordingWriter()
        unit = _build_unit(recording)

        unit.on_end()
        assert recording.flushed is True
        assert recording.closed is True

    def test_on_load_assigns_current_loss_and_skips_graph_when_not_first_session(self) -> None:
        """ on_load wires up the loss and skips graph writing off session 1 """
        recording = RecordingWriter()
        unit = _build_unit(recording)
        unit._session_id = 2
        unit._write_graph = False
        loop = _FakeLoop([_single_component()])

        unit.on_load(loop)
        assert unit._current_loss is loop.current_loss
        assert not recording.scalars

    def test_on_load_skips_graph_when_write_graph_disabled(self) -> None:
        """ A disabled write_graph flag skips graph tracing even on session 1 """
        recording = RecordingWriter()
        unit = _build_unit(recording)
        unit._session_id = 1
        unit._write_graph = False
        loop = _FakeLoop([_single_component()])

        unit.on_load(loop)
        assert unit._current_loss is loop.current_loss
        assert not recording.scalars

    def test_on_load_traces_graph_and_restores_training_mode(self) -> None:
        """ Graph is traced on session 1 and the model's training mode is restored afterwards """
        recording = RecordingWriter()
        unit = _build_unit(recording)
        unit._session_id = 1
        unit._write_graph = True
        loop = _FakeLoop([_single_component()], plugin=_TracerModule())

        unit.on_load(loop)

        assert len(recording.graphs) == 1
        traced_model, example_inputs = recording.graphs[0]
        assert isinstance(traced_model, nn.Module)

        def _contains_tensor(obj: object) -> bool:
            """ Recursively detect a Tensor within nested tuples/lists (matches graph args) """
            if isinstance(obj, torch.Tensor):
                return True
            if isinstance(obj, (tuple, list)):
                return any(_contains_tensor(o) for o in obj)
            return False

        assert example_inputs and _contains_tensor(example_inputs)
        assert loop.model.plugin.training is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

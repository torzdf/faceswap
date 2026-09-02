#!/usr/bin/env python3
# pylint:disable=protected-access,redefined-outer-name,too-few-public-methods,import-error
# pylint:disable=too-many-arguments,too-many-positional-arguments,unused-argument
# pylint:disable=missing-function-docstring
""" Unit tests for :mod:`lib.training.units.core.plugin_unit` """
from __future__ import annotations

import typing as T

import pytest
import torch

from lib.training.units.core.plugin_unit import PluginUnit


# =============================================================================
# Constants
# =============================================================================


CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda")

DEFAULT_FUNCTIONS = {"loss1": 1.0, "loss2": 2.0}

DEFAULT_CONFIG: dict[str, T.Any] = {"functions": DEFAULT_FUNCTIONS,
                                    "penalize_mask_loss": False,
                                    "eye_multiplier": 1.0,
                                    "mouth_multiplier": 1.0,
                                    "mask_loss": None}


# =============================================================================
# Fixtures
# =============================================================================

# Generic device-recording object


class RecordingObject:
    """ Minimal collaborator double that records every ``to(device)`` call

    Stands in for a tensor/model/loss and captures the devices it was moved onto so tests
    can assert on observable placement behavior without touching real tensors or GPUs
    """
    def __init__(self) -> None:
        self._seen_devices: list[T.Any] = []
        #: Whether the object participates in a computation graph before being detached.
        self.requires_grad: bool = True

    def to(self, device: T.Any) -> RecordingObject:
        """ Move onto ``device`` and record it; return ``self`` for chaining """
        self._seen_devices.append(device)
        return self

    @property
    def seen_devices(self) -> list[T.Any]:
        """ The devices this object has been moved onto, in call order """
        return self._seen_devices


# Batch / loader doubles


def _make_batch(inputs_per_batch: int = 3
                ) -> tuple[list[RecordingObject], list[RecordingObject], RecordingObject]:
    """ Build one ``(inputs, targets, meta)`` batch of :class:`RecordingObject` components """
    return ([RecordingObject() for _ in range(inputs_per_batch)],
            [RecordingObject() for _ in range(inputs_per_batch)],
            RecordingObject())


class FakeTrainLoader:
    """ Iterable TrainLoader double yielding ``(inputs, targets, meta)`` tuples.

    ``PluginUnit.step`` consumes the loader with ``next(self._loader)``, so each yielded item is a
    ``(inputs, targets, meta)`` tuple; every component is a :class:`RecordingObject` that records
    which device it was moved onto during the step
    """
    def __init__(self, inputs_per_batch: int = 3, num_batches: int = 10) -> None:
        self._batches: list[tuple[
            list[RecordingObject], list[RecordingObject], RecordingObject]] = [
            _make_batch(inputs_per_batch) for _ in range(num_batches)
        ]
        #: Index of the next batch to yield; PluginUnit.step advances it via next().
        self._index: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._batches):
            raise StopIteration()
        batch = self._batches[self._index]
        self._index += 1
        return batch


# Model / optimizer / trainer doubles


class FakeModel:
    """ Minimal FaceswapModel double exposing the ``state``/``info``/``plugin`` seams """
    class _Info:
        def __init__(self, output_shapes: T.Optional[list[T.Any]] = None) -> None:
            #: Nested per-face shapes so _configure_loss can compute smallest_output.
            self.output_shapes = output_shapes if output_shapes is not None else [
                [(4, 64)],
                [(8, 32)],
            ]

    class _Plugin:
        def __init__(self, is_rgb: bool = True) -> None:
            self.is_rgb = is_rgb

    def __init__(self, output_shapes: T.Optional[list[T.Any]] = None, is_rgb: bool = True) -> None:
        #: PluginUnit stores ``model.state`` here; no further behavior is required.
        self._state: object = object()
        self.info = FakeModel._Info(output_shapes)
        self.plugin = FakeModel._Plugin(is_rgb)

    @property
    def state(self):
        return self._state


class _StubState:
    """ Placeholder for :class:`~lib.model.plugin.State`; no persistence is tested """
    def __init__(self, plugin_name: str | None = None) -> None:
        self.name = plugin_name or "mock_plugin"


class FakeTrainer:
    """ Minimal TrainerPlugin double that returns a fixed number of losses per step """
    def __init__(self, num_losses: int = 3) -> None:
        #: ``on_load`` moves the trainer model onto the training device; record it here.
        self.model = RecordingObject()
        self.num_losses = num_losses

    def step(self, inputs: T.Any, targets: T.Any, meta: T.Any, loss_fn: T.Any, optimizer: T.Any):
        """ Return one fresh loss double per configured count """
        return [_StubLoss(num_losses=self.num_losses) for _ in range(self.num_losses)]


class _StubOptimizer:
    """ Placeholder optimizer; :class:`PluginUnit` only stores it and forwards it verbatim """
    def __init__(self, name: str | None = None) -> None:
        self.name = name or "mock_optimizer"


# Loss doubles


class _StubLoss:
    """ Double for a single loss value returned by the trainer during ``step()`` """
    def __init__(self, num_losses: int = 3) -> None:
        #: Number of losses the trainer is configured to return (kept for identification).
        self.num_losses = num_losses
        self.requires_grad: bool = True

    def detach(self):
        """ Detach from the computation graph; observable as ``requires_grad`` becoming False """
        self.requires_grad = False
        return self


class _StubLossCollator:
    """ Double for :class:`~lib.training.loss.LossCollator`.

    Records the configuration :class:`PluginUnit` forwards to it (via ``__call__``), records any
    device placement via ``to()``, and returns itself so a test can read back the exact wiring
    """
    def __init__(self) -> None:
        self.config: dict[str, T.Any] = {}
        self.devices_seen: list[T.Any] = []

    def __call__(self, **kwargs: T.Any) -> "_StubLossCollator":
        """ Record the construction arguments and return this instance """
        self.config.update(kwargs)
        return self

    def to(self, device: T.Any) -> "_StubLossCollator":
        """ Record placement onto ``device`` when the unit moves loss onto the training device """
        self.devices_seen.append(device)
        return self


TFactory = tuple[T.Callable[..., PluginUnit], _StubLossCollator]


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> TFactory:
    """ Build a PluginUnit with the real LossCollator patched to record forwarded config.

    Returns ``(build, recorder)`` where ``build(...)`` constructs a fully-isolated unit and
    ``recorder.config`` holds exactly what was forwarded to loss setup during that construction.
    The LossCollator patch is scoped to this test only (fresh instance per build call). """
    recorder = _StubLossCollator()
    monkeypatch.setattr("lib.training.units.core.plugin_unit.LossCollator", recorder)

    def build(
        config: dict[str, T.Any] | None = None,
        *,
        device: torch.device = CPU_DEVICE,
        loader=None,
        trainer=None,
        model=None,
        output_shapes=None,
        is_rgb: bool = True,
    ):
        cfg = dict(config or {})
        for key, default_value in DEFAULT_CONFIG.items():
            cfg.setdefault(key, default_value)
        loader = loader or FakeTrainLoader()
        trainer = trainer or FakeTrainer()
        optimizer = _StubOptimizer()
        model = model or FakeModel(
            output_shapes=output_shapes if output_shapes is not None else [
                [(4, 64)],
                [(8, 32)]],
            is_rgb=is_rgb)
        return PluginUnit(
            loader=loader, trainer=trainer, optimizer=optimizer, model=model, device=device,
            loss_functions=cfg["functions"], penalize_mask_loss=cfg["penalize_mask_loss"],
            eye_multiplier=cfg["eye_multiplier"], mouth_multiplier=cfg["mouth_multiplier"],
            mask_loss=cfg["mask_loss"])

    return build, recorder


# =============================================================================
# Initialization and Configuration
# =============================================================================


def test_current_loss_is_empty_before_step(factory: TFactory) -> None:
    """ ``current_loss`` reports an empty list until the first training step runs. """
    build, _ = factory
    unit = build()
    assert isinstance(unit.current_loss, list)
    assert len(unit.current_loss) == 0


def test_configure_loss_forwards_rgb_and_bgr(factory: TFactory) -> None:
    """ The composite loss receives ``color_order`` matching the model's channel layout. """
    build, recorder = factory
    for is_rgb, expected_color in [(True, "rgb"), (False, "bgr")]:
        build(is_rgb=is_rgb)
        assert recorder.config["color_order"] == expected_color
        # Functions and their weights are forwarded as parallel lists from the mapping.
        assert recorder.config["functions"] == list(DEFAULT_FUNCTIONS)
        assert list(recorder.config["weights"]) == list(DEFAULT_FUNCTIONS.values())


def test_configure_loss_respects_mask_settings(factory: TFactory) -> None:
    """ Mask penalties, multipliers and mask-loss type are forwarded verbatim to loss setup. """
    build, recorder = factory
    config = {
        "penalize_mask_loss": True,
        "eye_multiplier": 2.0,
        "mouth_multiplier": 3.0,
        "mask_loss": "mse",
    }
    build(config=config, output_shapes=[[(1, 2), (4, 5)]])
    assert recorder.config["use_mask"] is True
    assert recorder.config["eye_multiplier"] == 2.0
    assert recorder.config["mouth_multiplier"] == 3.0
    assert recorder.config["mask_loss"] == "mse"
    # smallest_output is the min of the smallest dimension across outputs whose first dim != 1.
    assert recorder.config["smallest_output"] == 5


def test_repr_reports_configuration(factory: TFactory) -> None:
    """ ``repr`` echoes class name and key configuration for debugging/training logs. """
    build, _ = factory
    unit = build()
    text = repr(unit)
    assert "PluginUnit" in text
    assert "loss_functions=" in text
    assert "penalize_mask_loss=False" in text
    assert "eye_multiplier=1.0" in text
    assert "mouth_multiplier=1.0" in text
    assert "mask_loss=None" in text


def test_repr_reflects_configuration_changes(factory: TFactory) -> None:
    """ ``repr`` updates when the applied configuration changes (e.g. eye multiplier). """
    build, _ = factory
    unit = build(config={"eye_multiplier": 0.5, "penalize_mask_loss": True})
    text = repr(unit)
    assert "eye_multiplier=0.5" in text
    assert "penalize_mask_loss=True" in text


# =============================================================================
# Step Processing / Batch-loss Tracking
# =============================================================================


def test_step_records_returned_losses(factory: TFactory) -> None:
    """ ``current_loss`` ends with one detached entry per loss the trainer returns. """
    build, _ = factory
    for num_losses in (1, 2, 3):
        unit = build(trainer=FakeTrainer(num_losses=num_losses))
        unit.step(42)
        assert len(unit.current_loss) == num_losses
        # Detach is observable: collected losses no longer require grad.
        for loss in unit.current_loss:
            assert loss.requires_grad is False


def test_step_does_not_accumulate_losses(factory: TFactory) -> None:
    """ Each step resets before recording; the count never grows across iterations. """
    build, _ = factory
    trainer = FakeTrainer(num_losses=3)
    unit = build(trainer=trainer)
    unit.step(1)
    assert len(unit.current_loss) == 3
    unit.step(2)
    assert len(unit.current_loss) == 3


def test_step_moves_batch_to_training_device(factory: TFactory) -> None:
    """ Every batch component is moved onto the configured device before training. """
    build, _ = factory
    for expected in (CPU_DEVICE.type, CUDA_DEVICE.type):
        device = CUDA_DEVICE if expected == "cuda" else CPU_DEVICE
        loader = FakeTrainLoader()
        unit = build(loader=loader, device=device)
        unit.step(1)
        inputs, targets, meta = loader._batches[0]
        # Every batch component is moved onto the configured device exactly once during the step.
        assert all(d.seen_devices == [device] for d in inputs)
        assert all(t.seen_devices == [device] for t in targets)
        assert meta.seen_devices == [device]


# =============================================================================
# Load / Device Placement
# =============================================================================


def test_on_load_moves_collaborators_to_device(factory: TFactory) -> None:
    """ ``on_load`` moves both the trainer model and loss onto the training device. """
    build, recorder = factory
    trainer = FakeTrainer()
    unit = build(trainer=trainer)
    unit.on_load(None)
    assert trainer.model.seen_devices == [CPU_DEVICE]
    assert recorder.devices_seen == [CPU_DEVICE]


# =============================================================================
# Capability Contract
# =============================================================================


def test_capability_flags_report_step_and_load(factory: TFactory) -> None:
    """ PluginUnit advertises its ``step`` and ``on_load`` capabilities to the training loop. """
    build, _ = factory
    unit = build()
    assert unit.has_step is True
    assert unit.has_load is True
    assert unit.has_save is False
    assert unit.has_update is False
    assert unit.has_end is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.gpu_stats._base` """
from __future__ import annotations

import typing as T

import pytest
from pytest_mock import MockerFixture

from lib.system.gpu_stats._base import BiggestGPUInfo, GPUInfo, _GPUStats

if T.TYPE_CHECKING:
    from pytest import LogCaptureFixture


# =============================================================================
# Fixtures
# =============================================================================

class _DummyData:  # pylint: disable=too-few-public-methods
    """ Dummy data for initializing and testing :class:`_GPUStats` """
    device_count: int = 2
    active_devices: list[int] = [0, 1]
    handles: list[int] = [0, 1]
    driver: str = "test_driver"
    device_names: list[str] = ["test_device_0", "test_device_1"]
    vram: list[int] = [1024, 2048]
    free_vram: list[int] = [512, 1024]


@pytest.fixture(name="gpu_stats_instance")
def fixture__gpu_stats_instance(mocker: MockerFixture) -> _GPUStats:
    """ Create a _GPUStats instance with all abstract methods mocked """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(_GPUStats, '_initialize')
    mocker.patch.object(_GPUStats, '_shutdown')
    mocker.patch.object(_GPUStats, '_get_device_count',
                        return_value=_DummyData.device_count)
    mocker.patch.object(_GPUStats, '_get_active_devices',
                        return_value=_DummyData.active_devices)
    mocker.patch.object(_GPUStats, '_get_handles',
                        return_value=_DummyData.handles)
    mocker.patch.object(_GPUStats, '_get_driver',
                        return_value=_DummyData.driver)
    mocker.patch.object(_GPUStats, '_get_device_names',
                        return_value=_DummyData.device_names)
    mocker.patch.object(_GPUStats, '_get_vram',
                        return_value=_DummyData.vram)
    mocker.patch.object(_GPUStats, '_get_free_vram',
                        return_value=_DummyData.free_vram)

    return _GPUStats()


# =============================================================================
# GPUInfo - construction
# =============================================================================

def test_gpuinfo_fields() -> None:
    """ Dataclass stores all five attributes correctly """
    info = GPUInfo(vram=[1024],
                   vram_free=[512],
                   driver="470",
                   devices=["GPU-0"],
                   devices_active=[0])
    assert info.vram == [1024]
    assert info.vram_free == [512]
    assert info.driver == "470"
    assert info.devices == ["GPU-0"]
    assert info.devices_active == [0]


# =============================================================================
# BiggestGPUInfo - construction
# =============================================================================

def test_biggest_gpuinfo_fields() -> None:
    """ Dataclass stores all four attributes correctly """
    info = BiggestGPUInfo(card_id=0, device="GPU-0", free=500.0, total=1024.0)
    assert info.card_id == 0
    assert info.device == "GPU-0"
    assert info.free == 500.0
    assert info.total == 1024.0


# =============================================================================
# _GPUStats - construction
# =============================================================================

def test_lifecycle_initialize_shutdown(gpu_stats_instance: _GPUStats) -> None:
    """ _is_initialized is False after init and shutdown """
    assert gpu_stats_instance._is_initialized is False


def test_all_attributes_populated(gpu_stats_instance: _GPUStats) -> None:
    """ Constructor stores results from all abstract methods """
    assert gpu_stats_instance._device_count == 2
    assert gpu_stats_instance._active_devices == [0, 1]
    assert gpu_stats_instance._handles == [0, 1]
    assert gpu_stats_instance._driver == "test_driver"
    assert gpu_stats_instance._device_names == ["test_device_0", "test_device_1"]
    assert gpu_stats_instance._vram == [1024, 2048]
    assert gpu_stats_instance._vram_free == [512, 1024]


def test_logger_is_set_when_log_true(gpu_stats_instance: _GPUStats) -> None:
    """ _logger is set when log=True """
    assert gpu_stats_instance._logger is not None


def test_log_false_does_not_crash(mocker: MockerFixture) -> None:
    """ Constructing with log=False does not raise an exception """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(_GPUStats, '_initialize')
    mocker.patch.object(_GPUStats, '_shutdown')
    mocker.patch.object(_GPUStats, '_get_device_count', return_value=0)
    mocker.patch.object(_GPUStats, '_get_active_devices', return_value=[])
    mocker.patch.object(_GPUStats, '_get_handles', return_value=[])
    mocker.patch.object(_GPUStats, '_get_driver', return_value="")
    mocker.patch.object(_GPUStats, '_get_device_names', return_value=[])
    mocker.patch.object(_GPUStats, '_get_vram', return_value=[])
    mocker.patch.object(_GPUStats, '_get_free_vram', return_value=[])

    instance = _GPUStats(log=False)
    assert instance._logger is None


# =============================================================================
# _GPUStats - _log
# =============================================================================

def test_log_true_messages_logged(caplog: LogCaptureFixture, mocker: MockerFixture) -> None:
    """ Internal messages appear in the logger when log=True """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(_GPUStats, '_initialize')
    mocker.patch.object(_GPUStats, '_shutdown')
    mocker.patch.object(_GPUStats, '_get_device_count', return_value=0)
    mocker.patch.object(_GPUStats, '_get_active_devices', return_value=[])
    mocker.patch.object(_GPUStats, '_get_handles', return_value=[])
    mocker.patch.object(_GPUStats, '_get_driver', return_value="")
    mocker.patch.object(_GPUStats, '_get_device_names', return_value=[])
    mocker.patch.object(_GPUStats, '_get_vram', return_value=[])
    mocker.patch.object(_GPUStats, '_get_free_vram', return_value=[])
    mocker.patch("lib.utils._FS_BACKEND", "cpu")

    caplog.set_level("DEBUG")
    _GPUStats(log=True)

    assert any("Initializing" in msg for msg in caplog.messages)
    assert any("Initialized" in msg for msg in caplog.messages)


def test_log_false_no_messages(caplog: LogCaptureFixture, mocker: MockerFixture) -> None:
    """ No messages appear in the logger when log=False """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(_GPUStats, '_initialize')
    mocker.patch.object(_GPUStats, '_shutdown')
    mocker.patch.object(_GPUStats, '_get_device_count', return_value=0)
    mocker.patch.object(_GPUStats, '_get_active_devices', return_value=[])
    mocker.patch.object(_GPUStats, '_get_handles', return_value=[])
    mocker.patch.object(_GPUStats, '_get_driver', return_value="")
    mocker.patch.object(_GPUStats, '_get_device_names', return_value=[])
    mocker.patch.object(_GPUStats, '_get_vram', return_value=[])
    mocker.patch.object(_GPUStats, '_get_free_vram', return_value=[])
    mocker.patch("lib.utils._FS_BACKEND", "cpu")

    caplog.set_level("DEBUG")
    _GPUStats(log=False)

    assert not any("Initializing" in msg for msg in caplog.messages)
    assert not any("Initialized" in msg for msg in caplog.messages)


# =============================================================================
# _GPUStats - device_count property
# =============================================================================

def test_returns_device_count(gpu_stats_instance: _GPUStats) -> None:
    """ Property returns the stored device count """
    assert gpu_stats_instance.device_count == 2


# =============================================================================
# _GPUStats - cli_devices property
# =============================================================================

def test_returns_index_name_strings(gpu_stats_instance: _GPUStats) -> None:
    """ Property formats each device as "index: name" """
    assert gpu_stats_instance.cli_devices == ["0: test_device_0", "1: test_device_1"]


# =============================================================================
# _GPUStats - exclude_all_devices property
# =============================================================================

def test_not_all_excluded(gpu_stats_instance: _GPUStats) -> None:
    """ Property is False when some devices are active """
    assert gpu_stats_instance.exclude_all_devices is False


def test_all_excluded(gpu_stats_instance: _GPUStats) -> None:
    """ Property is True when every device index is excluded """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.extend([0, 1])
    assert gpu_stats_instance.exclude_all_devices is True


# =============================================================================
# _GPUStats - sys_info property
# =============================================================================

def test_returns_gpuinfo(gpu_stats_instance: _GPUStats) -> None:
    """ Property returns a GPUInfo dataclass with current values """
    expected = GPUInfo(vram=[1024, 2048],
                       vram_free=[512, 1024],
                       driver="test_driver",
                       devices=["test_device_0", "test_device_1"],
                       devices_active=[0, 1])
    assert gpu_stats_instance.sys_info == expected


# =============================================================================
# _GPUStats - _get_active_devices
# =============================================================================

def test_filters_excluded_devices(mocker: MockerFixture) -> None:
    """ Default implementation excludes indices in _EXCLUDE_DEVICES """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(_GPUStats, '_initialize')
    mocker.patch.object(_GPUStats, '_shutdown')
    mocker.patch.object(_GPUStats, '_get_device_count', return_value=2)
    mocker.patch.object(_GPUStats, '_get_handles', return_value=[0, 1])
    mocker.patch.object(_GPUStats, '_get_driver', return_value="test_driver")
    mocker.patch.object(_GPUStats, '_get_device_names',
                        return_value=["test_device_0", "test_device_1"])
    mocker.patch.object(_GPUStats, '_get_vram', return_value=[1024, 2048])
    mocker.patch.object(_GPUStats, '_get_free_vram', return_value=[512, 1024])

    instance = _GPUStats()
    _EXCLUDE_DEVICES.append(0)
    active = instance._get_active_devices()
    assert active == [1]


# =============================================================================
# _GPUStats - get_card_most_free
# =============================================================================

def test_most_free_card(gpu_stats_instance: _GPUStats) -> None:
    """ Returns the active GPU with the most free VRAM """
    result = gpu_stats_instance.get_card_most_free()
    assert result.card_id == 1
    assert result.device == "test_device_1"
    assert result.free == 1024
    assert result.total == 2048


def test_no_active_devices(mocker: MockerFixture) -> None:
    """ Returns fallback info when no active GPUs are available """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(_GPUStats, '_initialize')
    mocker.patch.object(_GPUStats, '_shutdown')
    mocker.patch.object(_GPUStats, '_get_device_count', return_value=2)
    mocker.patch.object(_GPUStats, '_get_active_devices', return_value=[])
    mocker.patch.object(_GPUStats, '_get_handles', return_value=[])
    mocker.patch.object(_GPUStats, '_get_driver', return_value="")
    mocker.patch.object(_GPUStats, '_get_device_names', return_value=["GPU-0", "GPU-1"])
    mocker.patch.object(_GPUStats, '_get_vram', return_value=[1024, 2048])
    mocker.patch.object(_GPUStats, '_get_free_vram', return_value=[512, 1024])

    instance = _GPUStats()
    result = instance.get_card_most_free()
    expected = BiggestGPUInfo(card_id=-1,
                              device="No GPU devices found",
                              free=2048,
                              total=2048)
    assert result == expected

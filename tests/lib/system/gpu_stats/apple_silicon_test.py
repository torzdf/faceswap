#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.gpu_stats.apple_silicon` """
from __future__ import annotations

import typing as T

import pytest

from lib.system.gpu_stats.apple_silicon import AppleSiliconStats

if T.TYPE_CHECKING:
    from pytest import LogCaptureFixture


# =============================================================================
# Fixtures
# =============================================================================

class _AppleSiliconDummyData:  # pylint: disable=too-few-public-methods
    """ Dummy data for initializing and testing :class:`AppleSiliconStats` """
    driver_allocated_memory: int = 256 * 1024 * 1024
    virtual_memory_available: int = 8 * 1024 * 1024 * 1024


@pytest.fixture(name="apple_silicon_instance")
def fixture__apple_silicon_instance(mocker: pytest.MockerFixture) -> T.Any:
    """ Create an AppleSiliconStats instance with all hardware calls mocked """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = False

    mocker.patch.object(as_module, "os")
    as_module.os.reset_mock()  # pylint: disable=no-member

    mocker.patch.object(as_module, "torch")
    as_module.torch.device.return_value = mocker.MagicMock(type="mps")
    as_module.torch.mps.driver_allocated_memory.return_value = (
        _AppleSiliconDummyData.driver_allocated_memory)

    mocker.patch.object(as_module, "psutil")
    as_module.psutil.virtual_memory.return_value.available = (  # pylint: disable=no-member
        _AppleSiliconDummyData.virtual_memory_available)

    return as_module.AppleSiliconStats()


@pytest.fixture(name="apple_silicon_module")
def fixture__apple_silicon_module(mocker: pytest.MockerFixture) -> T.Any:
    """ Provide the apple_silicon module with mocked dependencies

    Parameters
    ----------
    mocker : :class:`pytest_mock.MockerFixture`
        Mocker for patching hardware-dependent calls

    Returns
    -------
    module
        The apple_silicon module with mocked os, torch, and psutil
    """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = False

    mocker.patch.object(as_module, "os")
    mocker.patch.object(as_module, "torch")
    as_module.torch.device.return_value = mocker.MagicMock(type="mps")
    as_module.torch.mps.driver_allocated_memory.return_value = (
        _AppleSiliconDummyData.driver_allocated_memory)

    mocker.patch.object(as_module, "psutil")

    return as_module


# =============================================================================
# AppleSiliconStats - _initialize
# =============================================================================

def test_initialize_returns_early_when_already_initialized(
        apple_silicon_instance: AppleSiliconStats) -> None:
    """ _initialize returns without re-init when _is_initialized is True """
    apple_silicon_instance._is_initialized = True
    apple_silicon_instance._initialize()


def test_initialize_sets_mps_devices(apple_silicon_instance: AppleSiliconStats) -> None:
    """ _mps_devices is set to [torch.device("mps")] after _initialize """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = False

    assert len(apple_silicon_instance._mps_devices) == 1
    assert apple_silicon_instance._mps_devices[0].type == "mps"


# =============================================================================
# AppleSiliconStats - _initialize_metal
# =============================================================================

def test_initialize_metal_sets_global_flag(apple_silicon_instance: AppleSiliconStats) -> None:
    """ _METAL_INITIALIZED is set to True """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = False

    apple_silicon_instance._initialize_metal()
    assert as_module._METAL_INITIALIZED is True


def test_initialize_metal_sets_display_env(
        apple_silicon_instance: AppleSiliconStats) -> None:
    """ os.environ["DISPLAY"] is set to ":0" """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = False

    apple_silicon_instance._initialize_metal()
    as_module.os.environ.__setitem__.assert_called_with(  # pylint: disable=no-member
        "DISPLAY", ":0")


def test_initialize_metal_tries_xquartz(apple_silicon_instance: AppleSiliconStats) -> None:
    """ os.system("open -a XQuartz") is called once """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = False
    as_module.os.system.reset_mock()

    apple_silicon_instance._initialize_metal()
    as_module.os.system.assert_called_once_with("open -a XQuartz")


def test_initialize_metal_returns_early_when_already_initialized(
        apple_silicon_instance: AppleSiliconStats) -> None:
    """ No side effects when _METAL_INITIALIZED is already True """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = True
    as_module.os.system.reset_mock()
    as_module.os.environ.reset_mock()  # pylint: disable=no-member

    apple_silicon_instance._initialize_metal()
    as_module.os.system.assert_not_called()
    as_module.os.environ.__setitem__.assert_not_called()  # pylint: disable=no-member


# =============================================================================
# AppleSiliconStats - _test_torch
# =============================================================================

@pytest.mark.usefixtures("apple_silicon_module")
def test_test_torch_raises_faceswap_error() -> None:
    """ RuntimeError from torch.mps.driver_allocated_memory raises FaceswapError """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    as_module._METAL_INITIALIZED = False
    as_module.torch.mps.driver_allocated_memory.side_effect = RuntimeError("test error")

    with pytest.raises(Exception) as exc_info:
        as_module.AppleSiliconStats()
    assert "test error" in str(exc_info.value)


# =============================================================================
# AppleSiliconStats - _get_device_count
# =============================================================================

def test_get_device_count(apple_silicon_instance: AppleSiliconStats) -> None:
    """ Returns len(_mps_devices) """
    assert apple_silicon_instance._get_device_count() == 1


# =============================================================================
# AppleSiliconStats - _get_handles
# =============================================================================

def test_get_handles(apple_silicon_instance: AppleSiliconStats) -> None:
    """ Returns list(range(device_count)) """
    assert apple_silicon_instance._get_handles() == [0]


# =============================================================================
# AppleSiliconStats - _get_driver
# =============================================================================

def test_get_driver(apple_silicon_instance: AppleSiliconStats) -> None:
    """ Returns "Not Applicable" """
    assert apple_silicon_instance._get_driver() == "Not Applicable"


# =============================================================================
# AppleSiliconStats - _get_device_names
# =============================================================================

def test_get_device_names(apple_silicon_instance: AppleSiliconStats) -> None:
    """ Returns [d.type for d in _mps_devices] """
    assert apple_silicon_instance._get_device_names() == ["mps"]


# =============================================================================
# AppleSiliconStats - _get_vram
# =============================================================================

def test_get_vram(apple_silicon_instance: AppleSiliconStats) -> None:
    """ Returns MB per device from torch driver_allocated_memory """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    result = apple_silicon_instance._get_vram()
    expected = int(as_module.torch.mps.driver_allocated_memory() / 1 / (1024 * 1024))
    assert result == [expected]


# =============================================================================
# AppleSiliconStats - _get_free_vram
# =============================================================================

def test_get_free_vram(apple_silicon_instance: AppleSiliconStats) -> None:
    """ Returns MB from psutil virtual_memory().available """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import apple_silicon as as_module
    result = apple_silicon_instance._get_free_vram()
    expected = int(as_module.psutil.virtual_memory().available / 1 / (1024 * 1024))
    assert result == [expected]


# =============================================================================
# AppleSiliconStats - exclude_devices
# =============================================================================

def test_exclude_devices_logs_warning(
        apple_silicon_instance: AppleSiliconStats,
        caplog: LogCaptureFixture) -> None:
    """ Logs warning, no error raised """
    apple_silicon_instance.exclude_devices([0, 1])
    assert any("Apple Silicon does not support" in msg for msg in caplog.messages)

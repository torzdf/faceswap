#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.gpu_stats.nvidia` """
from __future__ import annotations

import typing as T

import pytest

from lib.system.gpu_stats.nvidia import NvidiaStats


# =============================================================================
# Fixtures
# =============================================================================

class _NvidiaDummyData:  # pylint: disable=too-few-public-methods
    """ Dummy data for initializing and testing :class:`NvidiaStats` """
    device_count: int = 2
    driver: str = "535.129.03"
    device_names: list[str] = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4080"]
    memory_total: int = 12 * 1024 * 1024 * 1024  # 12 GB in bytes
    memory_free: int = 10 * 1024 * 1024 * 1024  # 10 GB in bytes
    vram_mb: float = 12288.0  # 12 GB in MB
    free_vram_mb: float = 10240.0  # 10 GB in MB


@pytest.fixture(name="nvidia_stats_instance")
def fixture__nvidia_stats_instance(
        mocker: pytest.MockerFixture) -> T.Any:
    """ Create a NvidiaStats instance with all pynvml calls mocked

    Parameters
    ----------
    mocker : :class:`pytest_mock.MockerFixture`
        Mocker for patching pynvml calls
    request : :class:`pytest.FixtureRequest`
        Pytest fixture request for access to fixture arguments

    Returns
    -------
    :class:`NvidiaStats`
        A NvidiaStats instance ready for testing
    """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(nv_module, "pynvml")
    mocker.patch.object(nv_module, "os")
    nv_module.pynvml.reset_mock()  # pylint: disable=no-member
    nv_module.os.reset_mock()  # pylint: disable=no-member

    # CUDA_VISIBLE_DEVICES must return None so _get_active_devices doesn't filter
    nv_module.os.environ.get.return_value = None

    nv_module.pynvml.nvmlDeviceGetCount.return_value = _NvidiaDummyData.device_count
    nv_module.pynvml.nvmlSystemGetDriverVersion.return_value = _NvidiaDummyData.driver

    dummy_handle = mocker.MagicMock()
    nv_module.pynvml.nvmlDeviceGetHandleByIndex.side_effect = (
        lambda i: [dummy_handle, dummy_handle][i])

    dummy_name = _NvidiaDummyData.device_names
    nv_module.pynvml.nvmlDeviceGetName.side_effect = lambda h: dummy_name[0]

    mem_info = mocker.MagicMock()
    mem_info.total = _NvidiaDummyData.memory_total
    mem_info.free = _NvidiaDummyData.memory_free
    nv_module.pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

    return nv_module.NvidiaStats()


@pytest.fixture(name="nvidia_module")
def fixture__nvidia_module(mocker: pytest.MockerFixture) -> T.Any:
    """ Provide the nvidia module with mocked pynvml dependencies

    Parameters
    ----------
    mocker : :class:`pytest_mock.MockerFixture`
        Mocker for patching pynvml calls

    Returns
    -------
    module
        The nvidia module with mocked pynvml
    """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module
    mocker.patch.object(nv_module, "pynvml")
    mocker.patch.object(nv_module, "os")
    nv_module.pynvml.reset_mock()  # pylint: disable=no-member
    nv_module.os.reset_mock()  # pylint: disable=no-member

    nv_module.os.environ.get.return_value = None

    nv_module.pynvml.nvmlDeviceGetCount.return_value = _NvidiaDummyData.device_count
    nv_module.pynvml.nvmlSystemGetDriverVersion.return_value = _NvidiaDummyData.driver

    dummy_handle = mocker.MagicMock()
    nv_module.pynvml.nvmlDeviceGetHandleByIndex.side_effect = (
        lambda i: [dummy_handle, dummy_handle][i])

    dummy_name = _NvidiaDummyData.device_names
    nv_module.pynvml.nvmlDeviceGetName.side_effect = lambda h: dummy_name[0]

    mem_info = mocker.MagicMock()
    mem_info.total = _NvidiaDummyData.memory_total
    mem_info.free = _NvidiaDummyData.memory_free
    nv_module.pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

    return nv_module


# =============================================================================
# NvidiaStats - _initialize
# =============================================================================

def test_init_no_reinit_when_active(nvidia_stats_instance: NvidiaStats) -> None:
    """ Does not re-initialize if already initialized """
    nvidia_stats_instance._is_initialized = True
    nvidia_stats_instance._initialize()


# =============================================================================
# NvidiaStats - _shutdown
# =============================================================================

def test_shutdown(nvidia_stats_instance: NvidiaStats) -> None:
    """ Calls pynvml.nvmlShutdown() and resets _is_initialized """
    nvidia_stats_instance._shutdown()
    assert nvidia_stats_instance._is_initialized is False


# =============================================================================
# NvidiaStats - _get_device_count
# =============================================================================

def test_get_device_count(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns the device count from pynvml """
    assert nvidia_stats_instance._get_device_count() == 2


# =============================================================================
# NvidiaStats - _get_active_devices
# =============================================================================

def test_active_filters_cuda(nvidia_stats_instance: NvidiaStats) -> None:
    """ Filters device indices by CUDA_VISIBLE_DEVICES """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module

    nv_module.os.environ.get.return_value = "0"
    try:
        result = nvidia_stats_instance._get_active_devices()
        assert result == [0]
    finally:
        nv_module.os.environ.get.return_value = None


# =============================================================================
# NvidiaStats - _get_handles
# =============================================================================

def test_get_handles(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns list of handles from pynvml """
    handles = nvidia_stats_instance._get_handles()
    assert len(handles) == 2


def test_handles_zero_dev(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns empty list when device_count is 0 """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module

    nv_module.pynvml.nvmlDeviceGetCount.return_value = 0
    nvidia_stats_instance._device_count = 0
    result = nvidia_stats_instance._get_handles()
    assert result == []


# =============================================================================
# NvidiaStats - _get_driver
# =============================================================================

def test_get_driver(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns the driver version from pynvml """
    assert nvidia_stats_instance._get_driver() == "535.129.03"


def test_drv_error_fallback(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns fallback string on NVMLError """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module

    err_cls = type("NVMLError", (Exception,), {})
    nv_module.pynvml.NVMLError = err_cls  # type: ignore[assignment]
    nv_module.pynvml.nvmlSystemGetDriverVersion.side_effect = (
        err_cls("test error"))
    try:
        result = nvidia_stats_instance._get_driver()
        assert result == "No Nvidia driver found"
    finally:
        nv_module.pynvml.nvmlSystemGetDriverVersion.side_effect = None
        nv_module.pynvml.nvmlSystemGetDriverVersion.return_value = (
            "535.129.03")


# =============================================================================
# NvidiaStats - _get_device_names
# =============================================================================

def test_get_device_names(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns list of GPU names from pynvml """
    names = nvidia_stats_instance._get_device_names()
    assert len(names) == 2


# =============================================================================
# NvidiaStats - _get_vram
# =============================================================================

def test_get_vram(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns VRAM in MB for each GPU """
    vram = nvidia_stats_instance._get_vram()
    assert len(vram) == 2
    assert vram[0] == 12288.0


# =============================================================================
# NvidiaStats - _get_free_vram
# =============================================================================

def test_get_free_vram(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns free VRAM in MB for each GPU """
    free = nvidia_stats_instance._get_free_vram()
    assert len(free) == 2
    assert free[0] == 10240.0


# =============================================================================
# NvidiaStats - exclude_devices
# =============================================================================

def test_excl_sets_cuda_env(nvidia_stats_instance: NvidiaStats) -> None:
    """ Sets CUDA_VISIBLE_DEVICES to active non-excluded devices """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    nvidia_stats_instance.exclude_devices([0])

    nv_module.os.environ.__setitem__.assert_called_with(  # pylint: disable=no-member
        "CUDA_VISIBLE_DEVICES", "1")


def test_excl_empty_noop(nvidia_stats_instance: NvidiaStats) -> None:
    """ Returns early when devices list is empty, no env set """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    nv_module.os.environ.__setitem__.reset_mock()  # pylint: disable=no-member
    nvidia_stats_instance.exclude_devices([])
    nv_module.os.environ.__setitem__.assert_not_called()  # pylint: disable=no-member


def test_excl_all_excluded(nvidia_stats_instance: NvidiaStats) -> None:
    """ Sets CUDA_VISIBLE_DEVICES to empty string when all devices excluded """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    nvidia_stats_instance.exclude_devices([0, 1])

    nv_module.os.environ.__setitem__.assert_called_with(  # pylint: disable=no-member
        "CUDA_VISIBLE_DEVICES", "")


def test_excl_logs_cuda_env(nvidia_stats_instance: NvidiaStats) -> None:
    """ Iterates over os.environ to find CUDA-related variables """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    nvidia_stats_instance.exclude_devices([0])

    nv_module.os.environ.items.assert_called()  # pylint: disable=no-member


def test_excl_preexisting(nvidia_stats_instance: NvidiaStats) -> None:
    """ Respects pre-existing _EXCLUDE_DEVICES entries """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import nvidia as nv_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    _EXCLUDE_DEVICES.append(1)
    nvidia_stats_instance.exclude_devices([0])

    nv_module.os.environ.__setitem__.assert_called_with(  # pylint: disable=no-member
        "CUDA_VISIBLE_DEVICES", "")

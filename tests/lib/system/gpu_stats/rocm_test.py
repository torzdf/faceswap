#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.gpu_stats.rocm` """
from __future__ import annotations

import os
import typing as T

import pytest
from pytest_mock import MockerFixture

from lib.system.gpu_stats.rocm import ROCm


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(name="rocm_instance")
def fixture__rocm_instance(mocker: MockerFixture) -> T.Any:
    """ Create a ROCm instance with no sysfs paths (no hardware)

    Parameters
    ----------
    mocker : :class:`pytest_mock.MockerFixture`
        Mocker for patching module dependencies

    Returns
    -------
    :class:`ROCm`
        A ROCm instance with empty sysfs paths ready for testing
    """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(rc_module, "torch")
    mocker.patch.object(rc_module, "os")
    rc_module.os.reset_mock()  # pylint: disable=no-member

    mocker.patch.object(rc_module, "which")
    mocker.patch.object(rc_module, "run")
    rc_module.which.return_value = None

    rc_module._is_wsl = False
    rc_module._sysfs_paths = []

    return rc_module.ROCm()


@pytest.fixture(name="sysfs_paths")
def fixture__sysfs_paths(mocker: MockerFixture) -> T.Any:
    """ Create a ROCm instance with mocked sysfs paths for AMD GPUs

    Parameters
    ----------
    mocker : :class:`pytest_mock.MockerFixture`
        Mocker for patching module dependencies

    Returns
    -------
    :class:`ROCm`
        A ROCm instance with mocked sysfs data ready for testing
    """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    mocker.patch.object(rc_module, "torch")
    mocker.patch.object(rc_module, "os")
    rc_module.os.reset_mock()  # pylint: disable=no-member

    mocker.patch.object(rc_module, "which")
    mocker.patch.object(rc_module, "run")
    rc_module.which.return_value = None

    rc_module._is_wsl = False
    rc_module._sysfs_paths = ["/sys/class/drm/card0/device", "/sys/class/drm/card1/device"]

    rc_module.os.path.basename = os.path.basename
    rc_module.os.path.exists.side_effect = lambda p: p == "/sys/class/drm/"
    rc_module.os.listdir.side_effect = (
        lambda p: ["card0", "card1"] if p == "/sys/class/drm/" else [])
    rc_module.os.path.join.side_effect = os.path.join
    rc_module.os.path.isdir.side_effect = lambda p: "device" in p
    rc_module.os.environ.get.return_value = None

    def sysfs_side_effect(path: str) -> str:  # pylint:disable=too-many-return-statements
        """Return controlled values for sysfs files."""
        filename = os.path.basename(path)
        if filename == "vendor":
            return "0x1002"
        if filename == "product_name":
            return "AMD Radeon RX 5700 XT"
        if filename == "product_number":
            return "0x6318"
        if filename == "device":
            return "0x6898"
        if filename == "mem_info_vram_total":
            return "8589934592"
        if filename == "mem_info_vram_used":
            return "1073741824"
        return ""

    rc_module.ROCm._from_sysfs_file = lambda self, path: sysfs_side_effect(path)

    return rc_module.ROCm()


def _get_filename(path: str) -> str:
    """Extract filename from a sysfs path."""
    return path.rstrip("/").split("/")[-1]


# =============================================================================
# ROCm - _from_sysfs_file
# =============================================================================

def test_from_sysfs_file_returns_content(
        sysfs_paths: ROCm) -> None:
    """ Returns file content stripped of whitespace """
    result = sysfs_paths._from_sysfs_file("/sys/class/drm/card0/device/vendor")
    assert result == "0x1002"


def test_from_sysfs_file_handles_missing_file(
        sysfs_paths: ROCm) -> None:
    """ Returns empty string when file does not exist """
    result = sysfs_paths._from_sysfs_file("/sys/class/drm/card0/nonexistent")
    assert result == ""


# =============================================================================
# ROCm - _get_sysfs_paths
# =============================================================================

def test_get_sysfs_paths_empty_when_no_sysfs(
        sysfs_paths: ROCm,
        mocker) -> None:
    """ Returns empty list when /sys/class/drm/ does not exist """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    # Re-patch os.path.exists to return False
    mocker.patch.object(rc_module.os.path, "exists", return_value=False)
    result = sysfs_paths._get_sysfs_paths()
    assert result == []


def test_get_sysfs_paths_filters_non_amd(
        sysfs_paths: ROCm) -> None:
    """ Only includes paths with AMD vendor ID 0x1002 """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = (
        lambda self, path: "0x1234" if _get_filename(path) == "vendor" else "")
    try:
        result = sysfs_paths._get_sysfs_paths()
        assert result == []
    finally:
        rc_module.ROCm._from_sysfs_file = original


# =============================================================================
# ROCm - _initialize
# =============================================================================

def test_initialize_sets_sysfs_paths(
        sysfs_paths: ROCm) -> None:
    """ _sysfs_paths is populated during initialization """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    assert sysfs_paths._sysfs_paths == rc_module._sysfs_paths  # pylint: disable=no-member


def test_initialize_returns_early_when_active(
        sysfs_paths: ROCm) -> None:
    """ Does not re-initialize if already initialized """
    sysfs_paths._is_initialized = True
    sysfs_paths._initialize()


# =============================================================================
# ROCm - _get_device_count
# =============================================================================

def test_get_device_count_returns_zero(
        rocm_instance: ROCm) -> None:
    """ Returns 0 when no sysfs paths are available """
    assert rocm_instance._get_device_count() == 0


# =============================================================================
# ROCm - _get_handles
# =============================================================================

def test_get_handles_returns_sysfs_paths(
        sysfs_paths: ROCm) -> None:
    """ Returns the list of sysfs paths """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    result = sysfs_paths._get_handles()
    assert result == rc_module._sysfs_paths  # pylint: disable=no-member


# =============================================================================
# ROCm - _get_driver
# =============================================================================

def test_get_driver_parses_modinfo(
        sysfs_paths: ROCm) -> None:
    """ Extracts version string from modinfo amdgpu output """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    rc_module.run.return_value.stdout = (  # pylint: disable=no-member
        "version: 5.15.0\nother: stuff")
    result = sysfs_paths._get_driver()
    assert result == "5.15.0"


def test_get_driver_handles_modinfo_error(
        sysfs_paths: ROCm) -> None:
    """ Returns empty string when modinfo fails """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    rc_module.run.side_effect = Exception("modinfo not found")
    result = sysfs_paths._get_driver()
    assert result == ""


def test_get_driver_returns_unknown_in_wsl(
        rocm_instance: ROCm) -> None:
    """ Returns 'unknown (wsl2)' when running in WSL """
    rocm_instance._is_wsl = True
    try:
        result = rocm_instance._get_driver()
        assert result == "unknown (wsl2)"
    finally:
        rocm_instance._is_wsl = False


# =============================================================================
# ROCm - _get_device_names
# =============================================================================

def test_get_device_names_from_product_metadata(
        sysfs_paths: ROCm) -> None:
    """ Reads product_name and product_number from sysfs """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = (
        lambda self, path: (
            "AMD Radeon RX 5700 XT" if _get_filename(path) == "product_name" else (
                "0x6318" if _get_filename(path) == "product_number" else (
                    "0x6898" if _get_filename(path) == "device" else ""))))
    try:
        result = sysfs_paths._get_device_names()
        assert len(result) == 2
        assert result[0] == "AMD Radeon RX 5700 XT 0x6318"
    finally:
        rc_module.ROCm._from_sysfs_file = original


def test_get_device_names_falls_back_to_lookup(
        sysfs_paths: ROCm) -> None:
    """ Falls back to _DEVICE_LOOKUP when product metadata is empty """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = (
        lambda self, path: (
            "0x6898" if _get_filename(path) == "device" else ""))
    try:
        result = sysfs_paths._get_device_names()
        assert "AMD Radeon HD 5800 Series" in result
    finally:
        rc_module.ROCm._from_sysfs_file = original


def test_get_device_names_handles_missing_device_id(
        sysfs_paths: ROCm) -> None:
    """ Returns 'Not found' when device ID cannot be read """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = lambda self, path: ""
    try:
        result = sysfs_paths._get_device_names()
        assert "Not found" in result
    finally:
        rc_module.ROCm._from_sysfs_file = original


def test_get_device_names_handles_bad_device_id(
        sysfs_paths: ROCm) -> None:
    """ Returns the raw device_id string when it cannot be parsed as hex """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = (
        lambda self, path: (
            "not_a_hex_id" if _get_filename(path) == "device" else ""))
    try:
        result = sysfs_paths._get_device_names()
        assert "not_a_hex_id" in result
    finally:
        rc_module.ROCm._from_sysfs_file = original


# =============================================================================
# ROCm - _get_active_devices
# =============================================================================

def test_active_filters_hip_env(
        sysfs_paths: ROCm) -> None:
    """ Filters device indices by HIP_VISIBLE_DEVICES """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    rc_module.os.environ.get.return_value = "0"
    try:
        result = sysfs_paths._get_active_devices()
        assert result == [0]
    finally:
        rc_module.os.environ.get.return_value = None


def test_active_returns_all_when_no_hip_env(
        sysfs_paths: ROCm) -> None:
    """ Returns all devices when HIP_VISIBLE_DEVICES is not set """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    rc_module.os.environ.get.return_value = None
    result = sysfs_paths._get_active_devices()
    assert result == [0, 1]


# =============================================================================
# ROCm - _get_vram
# =============================================================================

def test_get_vram_converts_bytes_to_mb(
        sysfs_paths: ROCm) -> None:
    """ Converts bytes from sysfs to megabytes """
    result = sysfs_paths._get_vram()
    assert result == [8192, 8192]


def test_get_vram_handles_parse_error(
        sysfs_paths: ROCm) -> None:
    """ Returns 0 MB when sysfs value cannot be parsed """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = (
        lambda self, path: (
            "not_a_number" if "total" in path else ""))
    try:
        result = sysfs_paths._get_vram()
        assert result == [0, 0]
    finally:
        rc_module.ROCm._from_sysfs_file = original


# =============================================================================
# ROCm - _get_free_vram
# =============================================================================

def test_get_free_vram_calculates_correctly(
        sysfs_paths: ROCm) -> None:
    """ Computes free = total - used, converted to MB """
    result = sysfs_paths._get_free_vram()
    assert result == [7168, 7168]


def test_get_free_vram_returns_zero_when_total_zero(
        sysfs_paths: ROCm) -> None:
    """ Returns 0 when total VRAM is 0 """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = (
        lambda self, path: "0" if "total" in path else "0")
    try:
        result = sysfs_paths._get_free_vram()
        assert result == [0, 0]
    finally:
        rc_module.ROCm._from_sysfs_file = original


def test_get_free_vram_handles_parse_error(
        sysfs_paths: ROCm) -> None:
    """ Returns total MB when used value cannot be parsed """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    original = rc_module.ROCm._from_sysfs_file
    rc_module.ROCm._from_sysfs_file = (
        lambda self, path: (
            "8589934592" if "total" in path else "not_a_number"))
    try:
        result = sysfs_paths._get_free_vram()
        assert result == [8192, 8192]
    finally:
        rc_module.ROCm._from_sysfs_file = original


# =============================================================================
# ROCm - exclude_devices
# =============================================================================

def test_excl_sets_hip_env(
        sysfs_paths: ROCm) -> None:
    """ Sets HIP_VISIBLE_DEVICES to active non-excluded devices """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    sysfs_paths.exclude_devices([0])

    rc_module.os.environ.__setitem__.assert_called_with(  # pylint: disable=no-member
        "HIP_VISIBLE_DEVICES", "1")


def test_excl_all_excluded(
        sysfs_paths: ROCm) -> None:
    """ Sets HIP_VISIBLE_DEVICES to empty string when all devices excluded """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    sysfs_paths.exclude_devices([0, 1])

    rc_module.os.environ.__setitem__.assert_called_with(  # pylint: disable=no-member
        "HIP_VISIBLE_DEVICES", "")


def test_excl_empty_noop(
        sysfs_paths: ROCm) -> None:
    """ Returns early when devices list is empty, no env set """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    rc_module.os.environ.__setitem__.reset_mock()  # pylint: disable=no-member
    sysfs_paths.exclude_devices([])
    rc_module.os.environ.__setitem__.assert_not_called()  # pylint: disable=no-member


def test_excl_logs_hip_env(
        sysfs_paths: ROCm) -> None:
    """ Iterates over os.environ to find HIP-related variables """
    # pylint:disable=import-outside-toplevel
    from lib.system.gpu_stats import rocm as rc_module
    from lib.system.gpu_stats._base import _EXCLUDE_DEVICES
    _EXCLUDE_DEVICES.clear()

    sysfs_paths.exclude_devices([0])

    rc_module.os.environ.items.assert_called()  # pylint: disable=no-member

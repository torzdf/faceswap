#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.gpu_stats.cpu` """
from __future__ import annotations

import pytest

from lib.system.gpu_stats.cpu import CPUStats


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(name="cpu_stats_instance")
def fixture__cpu_stats_instance() -> CPUStats:
    """ Create a :class:`CPUStats` instance with no hardware dependencies

    Returns
    -------
    :class:`CPUStats`
        A CPUStats instance ready for testing
    """
    return CPUStats()


# =============================================================================
# CPUStats - _get_device_count
# =============================================================================

def test_get_device_count(cpu_stats_instance: CPUStats) -> None:
    """ Always returns zero for CPU backends """
    assert cpu_stats_instance._get_device_count() == 0


# =============================================================================
# CPUStats - _get_handles
# =============================================================================

def test_get_handles(cpu_stats_instance: CPUStats) -> None:
    """ Returns an empty list for CPU backends """
    assert cpu_stats_instance._get_handles() == []


# =============================================================================
# CPUStats - _get_driver
# =============================================================================

def test_get_driver(cpu_stats_instance: CPUStats) -> None:
    """ Returns an empty string for CPU backends """
    assert cpu_stats_instance._get_driver() == ""


# =============================================================================
# CPUStats - _get_device_names
# =============================================================================

def test_get_device_names(cpu_stats_instance: CPUStats) -> None:
    """ Returns an empty list for CPU backends """
    assert cpu_stats_instance._get_device_names() == []


# =============================================================================
# CPUStats - _get_vram
# =============================================================================

def test_get_vram(cpu_stats_instance: CPUStats) -> None:
    """ Returns an empty list for CPU backends """
    assert cpu_stats_instance._get_vram() == []


# =============================================================================
# CPUStats - _get_free_vram
# =============================================================================

def test_get_free_vram(cpu_stats_instance: CPUStats) -> None:
    """ Returns an empty list for CPU backends """
    assert cpu_stats_instance._get_free_vram() == []


# =============================================================================
# CPUStats - exclude_devices
# =============================================================================

def test_exclude_devices_logs_warning(cpu_stats_instance: CPUStats, caplog: object) -> None:
    """ Logs warning, does not raise an error """
    cpu_stats_instance.exclude_devices([0, 1])
    assert any("CPU does not support excluding GPUs" in msg
               for msg in caplog.messages)


def test_exclude_devices_does_not_raise(cpu_stats_instance: CPUStats) -> None:
    """ Does not raise an exception """
    cpu_stats_instance.exclude_devices([0, 1])

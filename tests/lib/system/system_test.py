#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.system` """
from __future__ import annotations

import locale
import os
import platform
import sys
import typing as T
from subprocess import CalledProcessError
from unittest.mock import MagicMock

import pytest

from lib.system.system import (VALID_KERAS, VALID_PYTHON, VALID_TORCH,
                               _lines_from_command, Packages, System)

if T.TYPE_CHECKING:
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture


# =============================================================================
# Constants
# =============================================================================

class TestVersionConstants:
    """ Tests for the VALID_* version-constant tuples """
    def test_all_constants_are_version_tuples(self) -> None:
        """ each VALID_* is a (min, max) pair of int tuples with min <= max """
        for const in (VALID_PYTHON, VALID_TORCH, VALID_KERAS):
            assert isinstance(const, tuple) and len(const) == 2
            for version in const:
                assert isinstance(version, tuple) and len(version) == 2
                assert all(isinstance(part, int) for part in version)
            assert const[0] <= const[1]

    def test_known_boundaries(self) -> None:
        """ the constants hold their documented minimum/maximum versions """
        assert VALID_PYTHON == ((3, 11), (3, 13))
        assert VALID_TORCH == ((2, 3), (2, 12))
        assert VALID_KERAS == ((3, 14), (3, 14))


# =============================================================================
# Tier 1: _lines_from_command
# =============================================================================

class TestLinesFromCommand:
    """ Tests for :func:`_lines_from_command` """
    def test_returns_split_stdout(self, mocker: MockerFixture) -> None:
        """ success returns stdout split into lines """
        proc = MagicMock()
        proc.stdout = "  a \nb\n c  \n"
        mocker.patch("lib.system.system.run", return_value=proc)
        assert _lines_from_command(["echo", "hi"]) == ["  a ", "b", " c  "]

    def test_file_not_found_returns_empty(self, mocker: MockerFixture) -> None:
        """ a missing command yields an empty list """
        mocker.patch("lib.system.system.run", side_effect=FileNotFoundError)
        assert _lines_from_command(["nope"]) == []

    def test_process_error_returns_empty(self, mocker: MockerFixture) -> None:
        """ a failed command yields an empty list """
        mocker.patch("lib.system.system.run", side_effect=CalledProcessError(1, ["cmd"]))
        assert _lines_from_command(["cmd"]) == []


# =============================================================================
# Tier 2: System.__init__ and platform properties
# =============================================================================

class TestSystemInit:
    """ Tests for :meth:`System.__init__` """
    def test_attributes_match_platform(self) -> None:
        """ __init__ copies every public attribute from the running environment """
        instance = System()
        expected = ["platform", "system", "machine", "release", "processor",
                    "cpu_count", "python_implementation", "python_version",
                    "python_architecture", "encoding", "is_conda", "is_admin",
                    "is_virtual_env"]
        assert set(instance.__dict__) == set(expected)
        assert instance.platform == platform.platform()
        assert instance.system == platform.system().lower()
        assert instance.machine == platform.machine()
        assert instance.release == platform.release()
        assert instance.processor == platform.processor()
        assert instance.cpu_count == os.cpu_count()
        assert instance.python_implementation == platform.python_implementation()
        assert instance.python_version == platform.python_version()
        assert instance.python_architecture == platform.architecture()[0]
        assert instance.encoding == locale.getpreferredencoding()
        assert instance.is_conda == ("conda" in sys.version.lower() or
                                     os.path.exists(os.path.join(sys.prefix, "conda-meta")))
        assert isinstance(instance.is_admin, bool)
        assert isinstance(instance.is_virtual_env, bool)


class TestSystemPlatformProperties:
    """ Tests for :meth:`System.is_linux`, .is_macos and .is_windows """
    @staticmethod
    def _system(system: str) -> System:
        instance = object.__new__(System)
        instance.system = system
        return instance

    @pytest.mark.parametrize(("system", "linux", "macos", "windows"),
                             [("linux", True, False, False),
                              ("darwin", False, True, False),
                              ("windows", False, False, True)])
    def test_flags(self, system: str, linux: bool, macos: bool, windows: bool) -> None:
        """ each is_* property reflects the lowercased system value """
        instance = TestSystemPlatformProperties._system(system)
        assert instance.is_linux == linux
        assert instance.is_macos == macos
        assert instance.is_windows == windows


class TestSystemRepr:
    """ Tests for :meth:`System.__repr__` """
    def test_repr_public_only(self) -> None:
        """ repr shows the class name and public attributes only """
        instance = object.__new__(System)
        instance.platform = "Linux"
        instance.system = "linux"
        result = repr(instance)
        assert result.startswith("System(")
        assert "system=" in result
        instance._hidden = "x"
        assert "_hidden" not in repr(instance)


# =============================================================================
# Tier 1: System permission and virtual-env helpers
# =============================================================================

class TestGetPermissions:
    """ Tests for :meth:`System._get_permissions` """
    def test_non_windows_root(self, monkeypatch: MonkeyPatch) -> None:
        """ non-Windows returns True when uid is 0 """
        monkeypatch.setattr("lib.system.system.os.getuid", lambda: 0)
        instance = object.__new__(System)
        instance.system = "linux"
        assert instance._get_permissions()

    def test_non_windows_user(self, monkeypatch: MonkeyPatch) -> None:
        """ non-Windows returns False for a normal user """
        monkeypatch.setattr("lib.system.system.os.getuid", lambda: 1000)
        instance = object.__new__(System)
        instance.system = "linux"
        assert not instance._get_permissions()

    def test_windows_admin(self, mocker: MockerFixture) -> None:
        """ Windows returns True when IsUserAnAdmin is nonzero """
        ctypes_mock = MagicMock()
        ctypes_mock.windll.shell32.IsUserAnAdmin.return_value = 1
        mocker.patch("lib.system.system.ctypes", ctypes_mock)
        instance = object.__new__(System)
        instance.system = "windows"
        assert instance._get_permissions()

    def test_windows_not_admin(self, mocker: MockerFixture) -> None:
        """ Windows returns False when IsUserAnAdmin is zero """
        ctypes_mock = MagicMock()
        ctypes_mock.windll.shell32.IsUserAnAdmin.return_value = 0
        mocker.patch("lib.system.system.ctypes", ctypes_mock)
        instance = object.__new__(System)
        instance.system = "windows"
        assert not instance._get_permissions()


class TestCheckVirtualEnv:
    """ Tests for :meth:`System._check_virtual_env` """
    def test_conda_env(self, monkeypatch: MonkeyPatch) -> None:
        """ conda branch detects an 'envs' directory in the prefix parent """
        monkeypatch.setattr("lib.system.system.sys.prefix", "/opt/miniconda3/envs/myenv")
        instance = object.__new__(System)
        instance.is_conda = True
        assert instance._check_virtual_env()

    def test_conda_not_env(self, monkeypatch: MonkeyPatch) -> None:
        """ conda branch returns False when the parent is not 'envs' """
        monkeypatch.setattr("lib.system.system.sys.prefix", "/opt/miniconda3/bin")
        instance = object.__new__(System)
        instance.is_conda = True
        assert not instance._check_virtual_env()

    def test_venv(self, monkeypatch: MonkeyPatch) -> None:
        """ non-conda branch detects a venv via differing prefixes """
        monkeypatch.setattr("lib.system.system.sys.base_prefix", "/usr/lib/python3.11")
        monkeypatch.setattr("lib.system.system.sys.prefix", "/tmp/venv")
        instance = object.__new__(System)
        instance.is_conda = False
        assert instance._check_virtual_env()

    def test_plain_venv(self, monkeypatch: MonkeyPatch) -> None:
        """ non-conda branch returns False when base and current prefixes match """
        monkeypatch.setattr("lib.system.system.sys.base_prefix", "/usr")
        monkeypatch.setattr("lib.system.system.sys.prefix", "/usr")
        instance = object.__new__(System)
        instance.is_conda = False
        assert not instance._check_virtual_env()


# =============================================================================
# Tier 1: System.validate_python
# =============================================================================

class TestValidatePython:
    """ Tests for :meth:`System.validate_python` """
    def test_valid(self, monkeypatch: MonkeyPatch) -> None:
        """ a valid version within range returns True """
        instance = object.__new__(System)
        instance.python_architecture = "64bit"
        instance.python_version = "3.12.0"
        monkeypatch.setattr("lib.system.system.VALID_PYTHON", ((3, 11), (3, 13)))
        monkeypatch.setattr("lib.system.system.sys.version_info", (3, 12, 0))
        assert instance.validate_python()

    def test_max_version_param(self, monkeypatch: MonkeyPatch) -> None:
        """ max_version honours the supplied upper bound """
        instance = object.__new__(System)
        instance.python_architecture = "64bit"
        instance.python_version = "3.12.0"
        monkeypatch.setattr("lib.system.system.VALID_PYTHON", ((3, 11), (3, 13)))
        monkeypatch.setattr("lib.system.system.sys.version_info", (3, 12, 0))
        assert instance.validate_python(max_version=(3, 12))

    def test_32bit_exits(self, monkeypatch: MonkeyPatch) -> None:
        """ a 32-bit architecture causes SystemExit """
        instance = object.__new__(System)
        instance.python_architecture = "32bit"
        instance.python_version = "3.12.0"
        monkeypatch.setattr("lib.system.system.VALID_PYTHON", ((3, 11), (3, 13)))
        monkeypatch.setattr("lib.system.system.sys.version_info", (3, 12, 0))
        monkeypatch.setattr("builtins.input", lambda _: "")
        with pytest.raises(SystemExit):
            instance.validate_python()

    def test_below_min_exits(self, monkeypatch: MonkeyPatch) -> None:
        """ a version below the minimum causes SystemExit """
        instance = object.__new__(System)
        instance.python_architecture = "64bit"
        instance.python_version = "3.12.0"
        monkeypatch.setattr("lib.system.system.VALID_PYTHON", ((3, 11), (3, 13)))
        monkeypatch.setattr("lib.system.system.sys.version_info", (3, 10, 0))
        monkeypatch.setattr("builtins.input", lambda _: "")
        with pytest.raises(SystemExit):
            instance.validate_python()

    def test_above_max_exits(self, monkeypatch: MonkeyPatch) -> None:
        """ a version above the maximum causes SystemExit """
        instance = object.__new__(System)
        instance.python_architecture = "64bit"
        instance.python_version = "3.12.0"
        monkeypatch.setattr("lib.system.system.VALID_PYTHON", ((3, 11), (3, 13)))
        monkeypatch.setattr("lib.system.system.sys.version_info", (3, 14, 0))
        monkeypatch.setattr("builtins.input", lambda _: "")
        with pytest.raises(SystemExit):
            instance.validate_python()


# =============================================================================
# Tier 1: System.validate
# =============================================================================

_VALIDATE_CASES = (("other", "x86_64", False, True),
                   ("darwin", "arm64", True, False),
                   ("darwin", "arm64", False, True),
                   ("linux", "x86_64", True, False),
                   ("windows", "x86_64", True, False))
_VALIDATE_IDS = [f"{s}-{m}-conda={c}" for s, m, c, _ in _VALIDATE_CASES]


class TestValidate:
    """ Tests for :meth:`System.validate` """
    @pytest.mark.parametrize(("system", "machine", "is_conda", "should_exit"),
                             _VALIDATE_CASES, ids=_VALIDATE_IDS)
    def test_validate(self, mocker: MockerFixture, system: str, machine: str,
                      is_conda: bool, should_exit: bool) -> None:
        """ unsupported OS or Apple Silicon outside conda exits; supported systems pass """
        instance = object.__new__(System)
        instance.system = system
        instance.machine = machine
        instance.is_conda = is_conda
        mocker.patch.object(System, "validate_python")
        if should_exit:
            with pytest.raises(SystemExit):
                instance.validate()
        else:
            assert instance.validate() is None


# =============================================================================
# Tier 1: Packages.__init__ and property accessors
# =============================================================================

class TestPackagesInit:
    """ Tests for :meth:`Packages.__init__` """
    def test_which_and_attributes(self, mocker: MockerFixture) -> None:
        """ init resolves the conda path and initialises package containers """
        mocker.patch("lib.system.system.which", return_value="conda")
        mocker.patch("lib.system.system._lines_from_command", return_value=[])
        packages = Packages()
        assert packages._conda_exe == "conda"
        assert isinstance(packages._installed_python, dict)
        assert packages._installed_conda == ["Could not get Conda package list"]

    def test_no_conda(self, mocker: MockerFixture) -> None:
        """ missing conda leaves _installed_conda as the placeholder message """
        mocker.patch("lib.system.system.which", return_value=None)
        mocker.patch("lib.system.system._lines_from_command", return_value=[])
        packages = Packages()
        assert packages._conda_exe is None
        assert packages._installed_conda is None


class TestPackagesProperties:
    """ Tests for the Packages installed_python / installed_conda properties """
    def test_installed_python_property(self) -> None:
        """ installed_python returns the parsed dict verbatim """
        packages = object.__new__(Packages)
        packages._installed_python = {"numpy": "1.26.0"}
        assert packages.installed_python == {"numpy": "1.26.0"}

    def test_installed_conda_empty(self) -> None:
        """ installed_conda returns an empty dict when nothing was collected """
        packages = object.__new__(Packages)
        packages._installed_conda = None
        assert not packages.installed_conda

    def test_installed_conda_parsed(self) -> None:
        """ installed_conda maps names to (version, build, channel) tuples """
        packages = object.__new__(Packages)
        packages._installed_conda = [
            "# comment",
            "numpy             1.26.0           pypi_0              pypi",
            "pkg2     2.0b1      h78e105d_0       conda-forge",
        ]
        assert packages.installed_conda == {
            "numpy": ("1.26.0", "pypi_0", "pypi"),
            "pkg2": ("2.0b1", "h78e105d_0", "conda-forge"),
        }

    def test_installed_conda_pretty_placeholder(self) -> None:
        """ installed_conda_pretty returns the placeholder when nothing collected """
        packages = object.__new__(Packages)
        packages._installed_conda = None
        assert packages.installed_conda_pretty == "Could not get Conda package list"

    def test_installed_conda_pretty_lines(self) -> None:
        """ installed_conda_pretty joins the raw conda lines with newlines """
        packages = object.__new__(Packages)
        packages._installed_conda = ["numpy 1.0 pypi_0 pypi", "pip 24.0 pypi_0 pypi"]
        assert packages.installed_conda_pretty == "numpy 1.0 pypi_0 pypi\npip 24.0 pypi_0 pypi"

    def test_installed_python_pretty_aligned(self) -> None:
        """ installed_python_pretty right-pads keys so values align in one column """
        packages = object.__new__(Packages)
        packages._installed_python = {"ab": "1.0", "cde": "2.0"}
        pretty = packages.installed_python_pretty
        expected_col = max(len(k) for k in packages._installed_python) + 2
        lines = pretty.split("\n")
        assert len(lines) == 2
        for key, line in zip(sorted(packages._installed_python), lines):
            assert line.startswith(key.ljust(expected_col - 1))


# =============================================================================
# Tier 1: Packages.__repr__ and _get_installed_* parsers
# =============================================================================

class TestPackagesRepr:
    """ Tests for :meth:`Packages.__repr__` """
    def test_repr_public_properties_only(self) -> None:
        """ repr lists installed_python and installed_conda but hides private/pretty """
        packages = object.__new__(Packages)
        packages._installed_python = {"numpy": "1.0"}
        packages._installed_conda = ["numpy 1.0 pypi_0 pypi"]
        result = repr(packages)
        assert result.startswith("Packages(")
        assert "installed_python=" in result
        assert "installed_conda=" in result
        assert "_conda_exe" not in result
        assert "pretty" not in result


class TestGetInstalledPython:
    """ Tests for :meth:`Packages._get_installed_python` """
    def test_parses_pip_freeze(self, mocker: MockerFixture) -> None:
        """ parses 'name==version' lines, lowercases keys and skips malformed lines """
        mocker.patch("lib.system.system._lines_from_command", return_value=["numpy==1.26.0",
                                                                            "PACKAGE2==2.0.0",
                                                                            "# comment",
                                                                            "malformed-no-equals"])
        packages = object.__new__(Packages)
        assert packages._get_installed_python() == {"numpy": "1.26.0", "package2": "2.0.0"}

    def test_empty(self, mocker: MockerFixture) -> None:
        """ no output yields an empty dict """
        mocker.patch("lib.system.system._lines_from_command", return_value=[])
        packages = object.__new__(Packages)
        assert not packages._get_installed_python()


class TestGetInstalledConda:
    """ Tests for :meth:`Packages._get_installed_conda` """
    def test_no_conda_exe(self) -> None:
        """ collection is skipped when conda is not installed """
        packages = object.__new__(Packages)
        packages._conda_exe = None
        packages._installed_conda = ["old"]
        packages._get_installed_conda()
        assert packages._installed_conda == ["old"]

    def test_empty_output(self, mocker: MockerFixture) -> None:
        """ empty conda output records the placeholder message """
        mocker.patch("lib.system.system._lines_from_command", return_value=[])
        packages = object.__new__(Packages)
        packages._conda_exe = "conda"
        packages._get_installed_conda()
        assert packages._installed_conda == ["Could not get Conda package list"]

    def test_populated_output(self, mocker: MockerFixture) -> None:
        """ populated conda output is stored verbatim """
        lines = ["numpy 1.0 pypi_0 pypi", "pip 24.0 pypi_0 pypi"]
        mocker.patch("lib.system.system._lines_from_command", return_value=lines)
        packages = object.__new__(Packages)
        packages._conda_exe = "/usr/bin/conda"
        packages._get_installed_conda()
        assert packages._installed_conda == lines

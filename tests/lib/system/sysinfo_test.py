#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.sysinfo` """
from __future__ import annotations

import json
import typing as T

from collections import namedtuple
from io import StringIO
from unittest.mock import MagicMock

import pytest

from lib.gpu_stats import GPUInfo
from lib.system.sysinfo import _Configs, _State, _SysInfo, get_sysinfo

if T.TYPE_CHECKING:
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(name="sys_info_instance")
def sys_info_fixture(mocker: MockerFixture) -> _SysInfo:
    """ _SysInfo instance with all collaborators mocked """
    mocker.patch("lib.system.sysinfo.psutil")
    mocker.patch("lib.system.sysinfo.GPUStats",
                 return_value=MagicMock(sys_info=GPUInfo(vram=[],
                                                         vram_free=[],
                                                         driver="N/A",
                                                         devices=[],
                                                         devices_active=[])))
    mocker.patch("lib.system.sysinfo.git")
    mocker.patch("lib.system.sysinfo.Cuda")
    mocker.patch("lib.system.sysinfo.ROCm")
    system_mock = MagicMock()
    system_mock.is_conda = False
    system_mock.encoding = "utf-8"
    mocker.patch("lib.system.sysinfo.System", return_value=system_mock)
    packages_mock = MagicMock()
    packages_mock.installed_python_pretty = "Python 3.11"
    packages_mock.installed_conda_pretty = ""
    mocker.patch("lib.system.sysinfo.Packages", return_value=packages_mock)
    state_mock = MagicMock()
    state_mock.state_file = ""
    mocker.patch("lib.system.sysinfo._State", return_value=state_mock)
    configs_mock = MagicMock()
    configs_mock.configs = ""
    mocker.patch("lib.system.sysinfo._Configs", return_value=configs_mock)
    mocker.patch("lib.system.sysinfo.platform")
    return _SysInfo()


@pytest.fixture(name="configs_instance")
def configs_fixture(mocker: MockerFixture) -> _Configs:
    """ _Configs instance with filesystem access mocked """
    mocker.patch("lib.system.sysinfo.os.listdir", return_value=[])
    return _Configs()


@pytest.fixture(name="state_instance")
def state_fixture(mocker: MockerFixture) -> _State:
    """ _State instance with sys.argv mocked to non-training mode """
    mocker.patch("sys.argv", ["faceswap.py", "extract"])
    return _State()

# =============================================================================
# Tier 1: Pure / Low-level Functions
# =============================================================================


class TestFormatText:
    """ Tests for :meth:`_Configs._format_text` """
    def test_formats_key_value_pair(self) -> None:
        """ _format_text produces a consistently formatted key-value pair with aligned spacing """
        result = _Configs._format_text("backend", "pytorch")
        assert result == "backend:                  pytorch\n"

    def test_strips_whitespace(self) -> None:
        """ _format_text strips leading and trailing whitespace from both key and value """
        result = _Configs._format_text("  backend  ", "  pytorch  ")
        assert result == "backend:                  pytorch\n"

    def test_handles_empty_key(self) -> None:
        """ _format_text handles an empty key """
        result = _Configs._format_text("", "value")
        assert result == ":                         value\n"


class TestGetArg:
    """ Tests for :meth:`_State._get_arg` """
    @pytest.mark.parametrize("argv,short_opt,long_opt,expected",
                             [(["faceswap.py", "extract", "-m", "/path/to/model"],
                               "-m",
                               "--model-dir",
                               "/path/to/model"),
                              (["faceswap.py", "extract", "--model-dir", "/path/to/model"],
                               "-m",
                               "--model-dir",
                               "/path/to/model"),
                              (["faceswap.py", "extract"], "-m", "--model-dir", None),
                              (["faceswap.py", "extract", "-m"], "-m", "--model-dir", None)])
    def test_retrieves_arg_value(self,
                                 monkeypatch: MonkeyPatch,
                                 argv: list[str],
                                 short_opt: str,
                                 long_opt: str,
                                 expected: T.Optional[str]) -> None:
        """ _get_arg returns the value for a given command line option from sys.argv """
        monkeypatch.setattr("sys.argv", argv)
        result = _State._get_arg(short_opt, long_opt)
        assert result == expected


class TestParseIni:
    """ Tests for :meth:`_Configs._parse_ini` """
    def test_parses_ini_formatted(self, configs_instance: _Configs, monkeypatch: MonkeyPatch
                                  ) -> None:
        """ _parse_ini converts INI file content to a formatted string with key-value pairs """
        ini_content = ("[backend]\n"
                       "backend = pytorch\n"
                       "# comment\n"
                       "\n"
                       "learning_rate = 0.001\n")
        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(ini_content))

        result = configs_instance._parse_ini("dummy.ini")

        assert "backend:" in result
        assert "pytorch" in result
        assert "learning_rate:" in result
        assert "0.001" in result
        assert "# comment" not in result

    def test_skips_comments_and_blank_lines(self,
                                            configs_instance: _Configs,
                                            monkeypatch: MonkeyPatch) -> None:
        """ _parse_ini skips comment lines and blank lines """
        ini_content = "# header\n\n[section]\n# comment\n\nkey = value\n"
        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(ini_content))

        result = configs_instance._parse_ini("dummy.ini")

        assert "# header" not in result
        assert "# comment" not in result
        assert "section" in result
        assert "key:" in result
        assert "value" in result


class TestParseJson:
    """ Tests for :meth:`_Configs._parse_json` """
    def test_parses_json_formatted(self, configs_instance: _Configs, monkeypatch: MonkeyPatch
                                   ) -> None:
        """ _parse_json converts JSON file content to a formatted string with sorted keys """
        json_content = json.dumps({"backend": "pytorch", "learning_rate": "0.001"})
        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(json_content))

        result = configs_instance._parse_json("dummy.json")

        assert "backend:" in result
        assert "pytorch" in result
        assert "learning_rate:" in result
        assert "0.001" in result


# =============================================================================
# Tier 2: Class-Level Behavior
# =============================================================================

class TestConfigsInit:
    """ Tests for :meth:`_Configs.__init__` """
    def test_sets_configs_as_string(self, configs_instance: _Configs) -> None:
        """ _Configs __init__ sets configs as a string """
        assert isinstance(configs_instance.configs, str)

    def test_sets_config_dir_as_string(self, configs_instance: _Configs) -> None:
        """ _Configs __init__ sets config_dir as a string """
        assert isinstance(configs_instance.config_dir, str)


class TestConfigsGetConfigs:
    """ Tests for :meth:`_Configs._get_configs` """
    def test_empty_no_config_dir(self, mocker: MockerFixture) -> None:
        """ _get_configs returns empty string when config directory doesn't exist """
        mocker.patch("lib.system.sysinfo.os.listdir", side_effect=FileNotFoundError)
        configs = _Configs()
        result = configs._get_configs()
        assert result == ""


class TestConfigsParseConfigs:
    """ Tests for :meth:`_Configs._parse_configs` """
    def test_formats_configs(self, configs_instance: _Configs, monkeypatch: MonkeyPatch) -> None:
        """ _parse_configs parses config files and formats their content """
        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO("key = value\n"))
        result = configs_instance._parse_configs(["test.ini"])

        assert "key:" in result
        assert "value" in result


class TestStateIsTraining:
    """ Tests for :attr:`_State._is_training` """
    def test_true_train(self, monkeypatch: MonkeyPatch) -> None:
        """ _is_training returns True when sys.argv[1] is 'train' """
        monkeypatch.setattr("sys.argv", ["faceswap.py", "train"])
        state = _State()
        assert state._is_training is True

    def test_false_extract(self, monkeypatch: MonkeyPatch) -> None:
        """ _is_training returns False when sys.argv[1] is 'extract' """
        monkeypatch.setattr("sys.argv", ["faceswap.py", "extract"])
        state = _State()
        assert state._is_training is False

    def test_false_convert(self, monkeypatch: MonkeyPatch) -> None:
        """ _is_training returns False when sys.argv[1] is 'convert' """
        monkeypatch.setattr("sys.argv", ["faceswap.py", "convert"])
        state = _State()
        assert state._is_training is False


class TestStateGetStateFile:
    """ Tests for :meth:`_State._get_state_file` """
    def test_empty_not_training(self, monkeypatch: MonkeyPatch) -> None:
        """ _get_state_file returns empty string when not in training mode """
        monkeypatch.setattr("sys.argv", ["faceswap.py", "extract"])
        state = _State()
        assert state._get_state_file() == ""

    def test_empty_no_model_dir(self, monkeypatch: MonkeyPatch, mocker: MockerFixture) -> None:
        """ _get_state_file returns empty string when model_dir is None """
        monkeypatch.setattr("sys.argv", ["faceswap.py", "train"])
        mocker.patch.object(_State, "_get_arg", side_effect=[None, "trainer"])
        state = _State()
        assert state._get_state_file() == ""

    def test_empty_no_trainer(self, monkeypatch: MonkeyPatch, mocker: MockerFixture) -> None:
        """ _get_state_file returns empty string when trainer is None """
        monkeypatch.setattr("sys.argv", ["faceswap.py", "train"])
        mocker.patch.object(_State, "_get_arg", side_effect=["/path/to/model", None])
        state = _State()
        assert state._get_state_file() == ""

    def test_empty_no_state_file(self, monkeypatch: MonkeyPatch, mocker: MockerFixture) -> None:
        """ _get_state_file returns empty string when state file doesn't exist """
        monkeypatch.setattr("sys.argv",
                            ["faceswap.py", "train", "-m", "/path/to/model", "-t", "trainer"])
        mocker.patch.object(_State, "_get_arg", side_effect=["/path/to/model", "trainer"])
        mocker.patch("os.path.isfile", return_value=False)
        state = _State()
        assert state._get_state_file() == ""

    def test_returns_formatted_state(self, monkeypatch: MonkeyPatch, mocker: MockerFixture
                                     ) -> None:
        """ _get_state_file returns formatted state file content when all conditions are met """
        monkeypatch.setattr("sys.argv",
                            ["faceswap.py", "train", "-m", "/path/to/model", "-t", "trainer"])
        mocker.patch.object(_State, "_get_arg", side_effect=["/path/to/model", "trainer"])
        state_content = '{"epoch": 100, "loss": 0.001}'
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("builtins.open", lambda *args, **kwargs: StringIO(state_content))

        state = _State()
        result = state._get_state_file()

        assert "=============== State File =================" in result
        assert state_content in result


class TestStateInit:
    """ Tests for :meth:`_State.__init__` """
    def test_sets_state_file_as_string(self, state_instance: _State) -> None:
        """ _State __init__ sets state_file as a string """
        assert isinstance(state_instance.state_file, str)

    def test_empty_not_training_mode(self, state_instance: _State) -> None:
        """ _State __init__ sets state_file to empty string when not in training mode """
        assert state_instance.state_file == ""


# =============================================================================
# Tier 3: Integration / High-Level
# =============================================================================

class TestSysInfoInit:
    """ Tests for :meth:`_SysInfo.__init__` """
    def test_instantiates_without_error(self, sys_info_instance: _SysInfo) -> None:
        """ _SysInfo can be instantiated with mocked collaborators """
        assert isinstance(sys_info_instance, _SysInfo)


class TestSysInfoFormatRam:
    """ Tests for :meth:`_SysInfo._format_ram` """
    def test_formats_ram_in_megabytes(self, sys_info_instance: _SysInfo, monkeypatch: MonkeyPatch
                                      ) -> None:
        """ _format_ram formats RAM stats in megabytes """
        svmem = namedtuple("svmem", ["available", "free", "total", "used"])
        monkeypatch.setattr("lib.system.sysinfo.psutil",
                            MagicMock(virtual_memory=lambda: svmem(available=1073741824,
                                                                   free=536870912,
                                                                   total=2147483648,
                                                                   used=1073741824)))

        result = sys_info_instance._format_ram()

        assert "Total:" in result
        assert "Available:" in result
        assert "Used:" in result
        assert "Free:" in result
        assert "MB" in result

    def test_zero_mb_no_psutil(self) -> None:
        """ _format_ram returns 0MB for all RAM values when psutil is not installed """
        import lib.system.sysinfo as sysinfo_module  # pylint:disable=import-outside-toplevel

        original_psutil = sysinfo_module.psutil
        try:
            sysinfo_module.psutil = None
            instance = object.__new__(_SysInfo)
            result = instance._format_ram()
            assert "Total: 0MB" in result
            assert "Available: 0MB" in result
            assert "Used: 0MB" in result
            assert "Free: 0MB" in result
        finally:
            sysinfo_module.psutil = original_psutil


class TestSysInfoFullInfo:
    """ Tests for :meth:`_SysInfo.full_info` """
    def test_returns_all_sections(self, sys_info_instance: _SysInfo) -> None:
        """ full_info returns a string containing all expected sections """
        result = sys_info_instance.full_info()

        assert isinstance(result, str)
        assert "System Information" in result
        assert "Pip Packages" in result
        assert "Configs" in result

    def test_conda_section_included(self, sys_info_instance: _SysInfo, mocker: MockerFixture
                                    ) -> None:
        """ full_info includes Conda Packages section when running under Conda """
        mocker.patch.object(sys_info_instance._system, "is_conda", True)
        mock_conda_instance = MagicMock()
        mock_conda_instance.communicate.return_value = (b"conda 23.0.0\n", b"")
        mock_popen = MagicMock()
        mock_popen.__enter__ = MagicMock(return_value=mock_conda_instance)
        mock_popen.__exit__ = MagicMock(return_value=False)
        mocker.patch("lib.system.sysinfo.Popen", return_value=mock_popen)

        result = sys_info_instance.full_info()

        assert "Conda Packages" in result

    def test_conda_section_excluded(self, sys_info_instance: _SysInfo, mocker: MockerFixture
                                    ) -> None:
        """ full_info excludes Conda Packages section when not running under Conda """
        mocker.patch.object(sys_info_instance._system, "is_conda", False)

        result = sys_info_instance.full_info()

        assert "Conda Packages" not in result


class TestGetSysinfo:
    """ Tests for :func:`get_sysinfo` """
    def test_returns_full_info_string(self, mocker: MockerFixture) -> None:
        """ get_sysinfo returns the full_info string from _SysInfo """
        expected = "=== System Info ==="
        mocker.patch.object(_SysInfo, "full_info", return_value=expected)

        result = get_sysinfo()

        assert result == expected

    def test_re_raises_on_exception(self, mocker: MockerFixture) -> None:
        """ get_sysinfo re-raises exceptions from _SysInfo.full_info """
        mocker.patch.object(_SysInfo, "full_info", side_effect=RuntimeError("test error"))

        with pytest.raises(RuntimeError, match="test error"):
            get_sysinfo()

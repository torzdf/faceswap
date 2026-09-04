#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.system.ml_libs` """
from __future__ import annotations

import typing as T
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lib.system.ml_libs import (_Cuda, _Alternatives, CudaLinux, CudaWindows,
                                ROCm, get_cuda_finder)

if T.TYPE_CHECKING:
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(name="roc_instance")
def fixture__roc_instance(mocker: MockerFixture) -> ROCm:
    """ ROCm instance with system detection disabled for hermetic, GPU-free tests """
    mocker.patch.object(ROCm, "_rocm_check")
    return ROCm()


@pytest.fixture(name="cuda_linux_instance")
def fixture__cuda_linux_instance(mocker: MockerFixture) -> CudaLinux:
    """ CudaLinux instance with system detection disabled for hermetic, GPU-free tests """
    mocker.patch.object(_Cuda, "_get_versions")
    mocker.patch.object(_Cuda, "_get_version")
    mocker.patch.object(_Cuda, "_get_cudnn_versions")
    return CudaLinux()


# =============================================================================
# Tier 1: _Cuda version-tuple parsing
# =============================================================================

_CUDA_TUPLE_INPUTS = (("12.3", (12, 3)),
                      (".12.3", (12, 3)),
                      ("12.3.4", (12, 3)),
                      ("12", None),
                      ("a.b", None),
                      ("", None))
_CUDA_TUPLE_IDS = [f"v={i!r}" for i, _ in _CUDA_TUPLE_INPUTS]


class TestCudaTupleFromString:
    """ Tests for :meth:`_Cuda._tuple_from_string` """
    @pytest.mark.parametrize(("version", "expected"), _CUDA_TUPLE_INPUTS, ids=_CUDA_TUPLE_IDS)
    def test_parses_valid(self, version: str, expected: tuple[int, int] | None) -> None:
        """ _tuple_from_string parses a CUDA major.minor string into an int pair """
        assert _Cuda._tuple_from_string(version) == expected


# =============================================================================
# Tier 1: ROCm version parsing
# =============================================================================

_ROCM_TUPLE_INPUTS = (("6.0.1", (6, 0, 1)), ("6.0", None), ("a.b.c", None), ("6.0.1.2", None))
_ROCM_TUPLE_IDS = [f"v={i!r}" for i, _ in _ROCM_TUPLE_INPUTS]

_ROCM_STRING_INPUTS = (("rocm-6.0.1-extra", (6, 0, 1)),
                       ("v5.1.0-final", (5, 1, 0)),
                       ("no version here", None))
_ROCM_STRING_IDS = [f"s={i!r}" for i, _ in _ROCM_STRING_INPUTS]


class TestROCmVersionParsing:
    """ Tests for ROCm parsing :meth:`_tuple_from_string` and :meth:`_version_from_string` """
    @pytest.mark.parametrize(("version", "expected"), _ROCM_TUPLE_INPUTS, ids=_ROCM_TUPLE_IDS)
    def test_tuple_parses(self, version: str, expected: tuple[int, int, int] | None) -> None:
        """ _tuple_from_string parses a strict 'x.y.z' ROCm string into an int triple """
        assert ROCm._tuple_from_string(version) == expected

    @pytest.mark.parametrize(("string", "expected"), _ROCM_STRING_INPUTS, ids=_ROCM_STRING_IDS)
    def test_version_from_string(self, roc_instance: ROCm, string: str,
                                 expected: tuple[int, int, int] | None) -> None:
        """ _version_from_string extracts a 'x.y.z' ROCm version from surrounding text """
        assert roc_instance._version_from_string(string) == expected


# =============================================================================
# Tier 1: ROCm validation properties
# =============================================================================

class TestROCmValidationProperties:
    """ Tests for ROCm :attr:`valid_versions`, :attr:`valid_installed` and :attr:`is_valid` """
    def test_valid_versions_filtered(self, roc_instance: ROCm) -> None:
        """ valid_versions keeps only versions within the supported major.minor range """
        roc_instance.versions = [(6, 0, 1), (5, 0, 0), (6, 0, 3)]
        assert roc_instance.valid_versions == [(6, 0, 1), (6, 0, 3)]

    def test_valid_installed_flag(self, roc_instance: ROCm) -> None:
        """ valid_installed is True only when at least one installed version is supported """
        roc_instance.versions = [(5, 0, 0)]
        assert roc_instance.valid_installed is False
        roc_instance.versions = [(6, 0, 1)]
        assert roc_instance.valid_installed is True

    def test_is_valid_default(self, roc_instance: ROCm) -> None:
        """ is_valid is False when the default version sits outside the supported range """
        roc_instance.version = (0, 0, 0)
        assert roc_instance.is_valid is False
        roc_instance.version = (6, 0, 1)
        assert roc_instance.is_valid is True


# =============================================================================
# Tier 1: _Cuda file-based version parsing
# =============================================================================

class TestCudaFileBasedParsing:
    """ Tests for :meth:`version_from_version_file` and :meth:`cudnn_version_from_header` """
    def test_reads_version_json(self, cuda_linux_instance: CudaLinux, tmp_path: Path) -> None:
        """ version_from_version_file reads the CUDA major.minor from a folder's version.json """
        (tmp_path / "version.json").write_text('{"cuda_cudart": {"version": "12.4"}}',
                                               encoding="utf-8")
        assert cuda_linux_instance.version_from_version_file(str(tmp_path)) == (12, 4)

    def test_missing_version_file(self, cuda_linux_instance: CudaLinux, tmp_path: Path) -> None:
        """ version_from_version_file returns None when no version file is present """
        assert cuda_linux_instance.version_from_version_file(str(tmp_path)) is None

    def test_cudnn_header_parsed(self, cuda_linux_instance: CudaLinux, tmp_path: Path) -> None:
        """ cudnn_version_from_header parses MAJOR/MINOR/PATCHLEVEL into a triple """
        header = ("#define CUDNN_MAJOR 8\n"
                  "#define CUDNN_MINOR 6\n"
                  "#define CUDNN_PATCHLEVEL 3\n")
        (tmp_path / "cudnn_version.h").write_text(header, encoding="utf-8")
        assert cuda_linux_instance.cudnn_version_from_header(str(tmp_path)) == (8, 6, 3)

    def test_empty_cudnn_header(self, cuda_linux_instance: CudaLinux, tmp_path: Path) -> None:
        """ cudnn_version_from_header returns None when the header has no version defines """
        (tmp_path / "cudnn_version.h").write_text("#define SOMETHING_ELSE 1\n", encoding="utf-8")
        assert cuda_linux_instance.cudnn_version_from_header(str(tmp_path)) is None


# =============================================================================
# Tier 1: _Alternatives output parsing
# =============================================================================

class TestAlternativesParsing:
    """ Tests for :attr:`_Alternatives.alternatives` and :attr:`_Alternatives.default` """
    def test_alternatives_extracted(self) -> None:
        """ alternatives returns the path from each line flagged with a priority marker """
        instance = _Alternatives("cuda")
        instance._output = ["/usr/local/cuda-12.0 - /usr/local/cuda-12.0 (priority 710)",
                            "/usr/local/cuda-11.8 - /usr/local/cuda-11.8 (priority 710)"]
        assert instance.alternatives == ["/usr/local/cuda-12.0", "/usr/local/cuda-11.8"]

    def test_default_selected(self) -> None:
        """ default returns the path named by the 'link currently points to' marker """
        instance = _Alternatives("cuda")
        instance._output = ["Alternative for cuda:",
                            "link currently points to /usr/local/cuda-12.0",
                            "/usr/local/cuda-11.8 - /usr/local/cuda-11.8 (priority 700)"]
        assert instance.default == "/usr/local/cuda-12.0"

    def test_empty_output(self) -> None:
        """ empty alternatives output yields no paths and an empty default """
        instance = _Alternatives("cuda")
        instance._output = []
        assert instance.alternatives == []
        assert instance.default == ""


# =============================================================================
# Tier 1: CudaLinux path helpers
# =============================================================================

class TestCudaLinuxPathHelpers:
    """ Tests for :meth:`CudaLinux._parent_from_targets` """
    def test_parent_from_targets_path(self, cuda_linux_instance: CudaLinux) -> None:
        """ _parent_from_targets returns the folder preceding a 'targets' entry in a CUDA path """
        result = cuda_linux_instance._parent_from_targets(
            "/usr/local/cuda-12.0/targets/x86_64-linux/lib")
        assert result == "/usr/local/cuda-12.0"

    def test_no_targets_folder(self, cuda_linux_instance: CudaLinux) -> None:
        """ _parent_from_targets returns an empty string when no targets entry is present """
        assert cuda_linux_instance._parent_from_targets("/some/where/not/cuda") == ""


# =============================================================================
# Tier 1: __repr__ for the CUDA and ROCm families
# =============================================================================

class TestRepr:
    """ Tests for :meth:`_Cuda.__repr__` and :meth:`ROCm.__repr__` """
    def test_cuda_repr_public_only(self, cuda_linux_instance: CudaLinux) -> None:
        """ CUDA repr shows the class name and public attributes only """
        cuda_linux_instance.versions = [(12, 3)]
        result = repr(cuda_linux_instance)
        assert result.startswith("CudaLinux(")
        assert "versions=" in result
        assert "_folder_prefix" not in result

    def test_rocm_repr_public_only(self, roc_instance: ROCm) -> None:
        """ ROCm repr shows the class name and public attributes only """
        roc_instance.versions = [(6, 0, 1)]
        result = repr(roc_instance)
        assert result.startswith("ROCm(")
        assert "versions=" in result
        assert "_re_version" not in result


# =============================================================================
# Tier 1: get_cuda_finder platform selection
# =============================================================================

class TestGetCUDAFinder:
    """ Tests for :func:`get_cuda_finder` """
    def test_returns_linux_finder(self, monkeypatch: MonkeyPatch) -> None:
        """ get_cuda_finder returns the Linux finder class on non-Windows platforms """
        fake = MagicMock()
        fake.system.return_value = "Linux"
        monkeypatch.setattr("lib.system.ml_libs.platform", fake)
        assert get_cuda_finder() is CudaLinux

    def test_returns_windows_finder(self, monkeypatch: MonkeyPatch) -> None:
        """ get_cuda_finder returns the Windows finder class when running on Windows """
        fake = MagicMock()
        fake.system.return_value = "Windows"
        monkeypatch.setattr("lib.system.ml_libs.platform", fake)
        assert get_cuda_finder() is CudaWindows

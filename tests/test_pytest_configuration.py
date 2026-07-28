# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for repository-wide pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def _root_conftest_module(config: pytest.Config) -> ModuleType:
    """Return the loaded root test configuration module."""
    expected_path = (config.rootpath / "tests" / "conftest.py").resolve()
    for plugin in config.pluginmanager.get_plugins():
        plugin_path = getattr(plugin, "__file__", None)
        if plugin_path is not None and Path(plugin_path).resolve() == expected_path:
            assert isinstance(plugin, ModuleType)
            return plugin
    raise AssertionError(f"Root conftest plugin not found at {expected_path}")


def _make_pytest_ini(pytester: pytest.Pytester) -> None:
    """Register the GPU marker in pytester's isolated test environment."""
    pytester.makeini(
        """
        [pytest]
        markers =
            requires_gpu: Test needs CUDA hardware
        """
    )


def test_requires_gpu_is_skipped_without_cuda(
    pytester: pytest.Pytester,
    pytestconfig: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_is_available = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=cuda_is_available)))
    _make_pytest_ini(pytester)
    pytester.makepyfile(
        test_gpu="""
            import pytest

            pytestmark = pytest.mark.requires_gpu

            def test_module_marked_gpu():
                raise AssertionError("GPU test ran without CUDA")

            @pytest.mark.requires_gpu
            def test_function_marked_gpu():
                raise AssertionError("GPU test ran without CUDA")
        """,
        test_cpu="""
            def test_cpu():
                pass
        """,
    )

    result = pytester.runpytest_inprocess(
        "-p",
        "no:cov",
        plugins=[_root_conftest_module(pytestconfig)],
    )

    result.assert_outcomes(passed=1, skipped=2)
    cuda_is_available.assert_called_once_with()


def test_cuda_is_not_probed_without_gpu_tests(
    pytester: pytest.Pytester,
    pytestconfig: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_is_available = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=cuda_is_available)))
    _make_pytest_ini(pytester)
    pytester.makepyfile(
        """
        def test_cpu():
            pass
        """
    )

    result = pytester.runpytest_inprocess(
        "-p",
        "no:cov",
        plugins=[_root_conftest_module(pytestconfig)],
    )

    result.assert_outcomes(passed=1)
    cuda_is_available.assert_not_called()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool(root_path: Path) -> ModuleType:
    path = root_path / "tools" / "discover_cuda_indexes.py"
    spec = importlib.util.spec_from_file_location("discover_cuda_indexes", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def discovery_tool(pytestconfig: pytest.Config) -> ModuleType:
    return _load_tool(pytestconfig.rootpath)


def _write_pyproject(tmp_path: Path, indexes: list[tuple[str, str]]) -> Path:
    blocks = "\n".join(
        f'[[tool.uv.index]]\nname = "{name}"\nurl = "{url}"\nexplicit = true\n' for name, url in indexes
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(blocks, encoding="utf-8")
    return path


def test_discovers_name_only_match(discovery_tool: ModuleType, tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, [("pytorch-cu129", "https://example.invalid/pytorch")])

    assert discovery_tool.discover_cuda_index_urls(path, "cu129", 1) == ["https://example.invalid/pytorch"]


def test_discovers_url_only_match(discovery_tool: ModuleType, tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, [("pytorch-nightly", "https://example.invalid/whl/cu129")])

    assert discovery_tool.discover_cuda_index_urls(path, "cu129", 1) == [
        "https://example.invalid/whl/cu129"
    ]


def test_deduplicates_matches(discovery_tool: ModuleType, tmp_path: Path) -> None:
    url = "https://example.invalid/whl/cu129"
    path = _write_pyproject(tmp_path, [("pytorch-cu129", url), ("mirror-cu129", url)])

    assert discovery_tool.discover_cuda_index_urls(path, "cu129", 1) == [url]


def test_partial_discovery_fails_below_minimum(
    discovery_tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_pyproject(
        tmp_path,
        [
            ("pytorch-cu129", "https://example.invalid/pytorch"),
            ("flashinfer", "https://example.invalid/flashinfer/cu129"),
            ("unrelated", "https://example.invalid/cpu"),
        ],
    )

    assert discovery_tool.main([str(path), "cu129", "3"]) == 1
    stderr = capsys.readouterr().err
    assert "expected at least 3 cu129 indexes" in stderr
    assert "found 2" in stderr

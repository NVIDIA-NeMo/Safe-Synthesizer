# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast

import pytest


class DiffLockfileTool(Protocol):
    """Typed interface exposed by the dynamically loaded lockfile tool."""

    def parse_packages(self, content: str) -> dict[str, object]: ...


def _load_tool(tool_path: Path) -> DiffLockfileTool:
    spec = importlib.util.spec_from_file_location("diff_lockfile", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(DiffLockfileTool, module)


@pytest.fixture(scope="module")
def diff_lockfile_tool(pytestconfig: pytest.Config) -> DiffLockfileTool:
    """Load the lockfile helper from pytest's discovered repository root."""
    return _load_tool(Path(pytestconfig.rootpath) / "tools" / "diff-lockfile.py")


def test_parse_packages_preserves_same_source_versions(diff_lockfile_tool: DiffLockfileTool) -> None:
    content = """
version = 1

[[package]]
name = "cuda-python"
version = "12.9.4"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "cuda-python"
version = "13.1.1"
source = { registry = "https://pypi.org/simple" }
"""

    packages = diff_lockfile_tool.parse_packages(content)

    assert set(packages) == {
        "cuda-python@https://pypi.org/simple@12.9.4",
        "cuda-python@https://pypi.org/simple@13.1.1",
    }

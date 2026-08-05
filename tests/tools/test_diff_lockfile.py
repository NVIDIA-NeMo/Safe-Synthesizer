# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast

import pytest
from packaging.version import Version


class Package(Protocol):
    """Typed package snapshot exposed by the dynamically loaded tool."""

    name: str
    version: Version
    source: str


class PackageChange(Protocol):
    """Typed package delta exposed by the dynamically loaded tool."""

    change: str
    old: Package | None
    new: Package | None


class LockfileDiff(Protocol):
    """Typed lockfile diff exposed by the dynamically loaded tool."""

    root: list[PackageChange]


class DiffLockfileTool(Protocol):
    """Typed interface exposed by the dynamically loaded lockfile tool."""

    def parse_packages(self, content: str) -> dict[str, Package]: ...

    def diff_packages(self, base: dict[str, Package], head: dict[str, Package], ref: str) -> LockfileDiff: ...


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


def test_diff_packages_classifies_changed_duplicate_variant(diff_lockfile_tool: DiffLockfileTool) -> None:
    base = diff_lockfile_tool.parse_packages(
        """
version = 1

[[package]]
name = "cuda-python"
version = "12.9.4"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "cuda-python"
version = "13.1.1"
source = { registry = "https://pypi.org/simple" }
""",
    )
    head = diff_lockfile_tool.parse_packages(
        """
version = 1

[[package]]
name = "cuda-python"
version = "12.9.4"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "cuda-python"
version = "13.2.0"
source = { registry = "https://pypi.org/simple" }
""",
    )

    diff = diff_lockfile_tool.diff_packages(base, head, ref="base..head")

    assert len(diff.root) == 1
    change = diff.root[0]
    assert change.change == "upgraded"
    assert change.old is not None and change.old.version == Version("13.1.1")
    assert change.new is not None and change.new.version == Version("13.2.0")

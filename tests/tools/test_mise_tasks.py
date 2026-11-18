# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import cast

import pytest
import yaml

QUALITY_TASKS = {
    "check",
    "check:format",
    "check:license:headers",
    "check:lint",
    "check:lock",
    "check:tasks",
    "check:type",
    "format",
    "lock:update",
}
REMOVED_QUALITY_TASKS = {"ci", "format-check", "lock-check", "typecheck", "validate"}


def _run_mise(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["mise", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _mise_tasks(repo_root: Path) -> dict[str, dict[str, object]]:
    result = _run_mise(repo_root, "tasks", "--json")
    assert result.returncode == 0, result.stderr
    return {task["name"]: task for task in json.loads(result.stdout)}


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """A directory with a uv that prints its arguments and a uname that prints $FAKE_UNAME."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("uv", "printf '%s\\n' \"$@\""), ("uname", "printf '%s\\n' \"$FAKE_UNAME\"")):
        tool = bin_dir / name
        tool.write_text(f"#!/usr/bin/env bash\n{body}\n")
        tool.chmod(0o755)
    return bin_dir


def _sync_dependencies(repo_root: Path, fake_bin: Path, profile: str | None, *, system: str = "Linux") -> list[str]:
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "FAKE_UNAME": system}
    env.pop("usage_profile", None)
    if profile is not None:
        env["usage_profile"] = profile

    result = subprocess.run(
        ["bash", str(repo_root / ".mise/tasks/deps/sync")],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


@pytest.mark.parametrize(
    ("profile", "system", "expected"),
    [
        (None, "Linux", []),
        ("dev", "Linux", []),
        ("all", "Linux", ["--all-groups", "--all-extras", "--no-extra", "cpu", "--no-extra", "cu129"]),
        ("all", "Darwin", ["--all-groups", "--all-extras", "--no-extra", "cu129", "--no-extra", "cu130"]),
        ("ci", "Linux", ["--all-groups", "--all-extras", "--no-extra", "cu129", "--no-extra", "cu130"]),
        ("engine", "Linux", ["--extra", "engine"]),
        ("cpu", "Linux", ["--extra", "cpu", "--extra", "engine"]),
        ("cu129", "Linux", ["--extra", "cu129", "--extra", "engine"]),
        ("cu130", "Linux", ["--extra", "cu130", "--extra", "engine"]),
    ],
)
def test_deps_sync_profile_arguments(
    pytestconfig: pytest.Config, fake_bin: Path, profile: str | None, system: str, expected: list[str]
) -> None:
    args = _sync_dependencies(Path(pytestconfig.rootpath), fake_bin, profile, system=system)

    assert args == ["sync", "--locked", *expected]


def test_deps_sync_rejects_unknown_profile(pytestconfig: pytest.Config, fake_bin: Path) -> None:
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "usage_profile": "cuda"}

    result = subprocess.run(
        ["bash", str(Path(pytestconfig.rootpath) / ".mise/tasks/deps/sync")],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "unknown profile 'cuda'" in result.stderr
    assert result.stdout == ""


def test_quality_task_contract(pytestconfig: pytest.Config) -> None:
    tasks = _mise_tasks(Path(pytestconfig.rootpath))

    assert QUALITY_TASKS <= tasks.keys()
    assert REMOVED_QUALITY_TASKS.isdisjoint(tasks)
    aliases = {alias for task in tasks.values() for alias in cast(list[str], task["aliases"])}
    assert "ci" not in aliases
    check_dependencies = cast(list[str], tasks["check"]["depends"])
    assert set(check_dependencies) == QUALITY_TASKS - {"check", "format", "lock:update"}


def test_public_tasks_are_valid_and_described(pytestconfig: pytest.Config) -> None:
    repo_root = Path(pytestconfig.rootpath)
    result = _run_mise(repo_root, "tasks", "validate")

    assert result.returncode == 0, result.stderr
    assert all(task["description"] for task in _mise_tasks(repo_root).values())


def test_quality_task_commands_preserve_check_contract(pytestconfig: pytest.Config) -> None:
    repo_root = Path(pytestconfig.rootpath)
    with (repo_root / ".mise/tasks/quality.toml").open("rb") as quality_file:
        tasks = tomllib.load(quality_file)

    lock_commands = tasks["check:lock"]["run"]
    assert isinstance(lock_commands, list)
    generator_command, uv_lock_command = lock_commands
    for token in ("gen_cuda_deps.py", "cuda_deps.toml", "--pyproject", "--installer", "--check"):
        assert token in generator_command
    assert "uv" in uv_lock_command and "lock" in uv_lock_command and "--check" in uv_lock_command
    assert "uv" in tasks["lock:update"]["run"] and "lock" in tasks["lock:update"]["run"]


def test_lock_hook_uses_read_only_task_for_both_inputs(pytestconfig: pytest.Config) -> None:
    config = (Path(pytestconfig.rootpath) / ".pre-commit-config.yaml").read_text()
    hook = config.split("      - id: uv-lock", maxsplit=1)[1].split("      - id:", maxsplit=1)[0]

    assert "entry: mise run check:lock" in hook
    assert "pyproject\\.toml" in hook
    assert "cuda_deps\\.toml" in hook


def test_local_gate_composes_check_and_test(pytestconfig: pytest.Config) -> None:
    result = _run_mise(Path(pytestconfig.rootpath), "run", "--dry-run", "check", ":::", "test")

    assert result.returncode == 0, result.stderr


def test_cuda_dependency_sources_route_to_dependency_and_quality_jobs(pytestconfig: pytest.Config) -> None:
    """Source-of-truth changes must take the same CI route as generated dependency metadata."""
    root = Path(pytestconfig.rootpath)
    action = yaml.safe_load((root / ".github/actions/detect-changes/action.yml").read_text())
    filters = action["runs"]["steps"][0]["with"]["filters"]
    outputs = action["outputs"]

    assert "cuda_deps.toml" in filters
    assert "constraints.txt" in filters
    for output in ("deps", "src_test_deps", "any"):
        assert "cuda_deps" in outputs[output]["value"]
        assert "constraints" in outputs[output]["value"]

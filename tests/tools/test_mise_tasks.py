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
def fake_uv(tmp_path: Path) -> Path:
    uv = tmp_path / "uv"
    uv.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    uv.chmod(0o755)
    return uv


def _sync_dependencies(
    repo_root: Path,
    fake_uv: Path,
    profile: str,
    *,
    python_version: str | None = None,
    python_override: str | None = None,
) -> list[str]:
    command = 'source "$1"; sync_nss_dependencies "$2" "${3:-}"'
    env = os.environ | {"MISE_CONFIG_ROOT": str(repo_root), "NSS_UV_BIN": str(fake_uv)}
    if python_version is not None:
        env["PYTHON_VERSION"] = python_version
    else:
        env.pop("PYTHON_VERSION", None)

    result = subprocess.run(
        ["bash", "-c", command, "bash", str(repo_root / ".mise/tasks/_lib.sh"), profile, python_override or ""],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


@pytest.mark.parametrize("python_version", ["3.11", "3.12", "3.13"])
def test_sync_dependencies_passes_requested_python(
    pytestconfig: pytest.Config, fake_uv: Path, python_version: str
) -> None:
    args = _sync_dependencies(Path(pytestconfig.rootpath), fake_uv, "dev", python_version=python_version)

    assert args == ["sync", "--frozen", "--python", python_version, "--group", "dev"]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("dev", ["--group", "dev"]),
        ("engine", ["--extra", "engine", "--group", "dev"]),
        ("cpu", ["--extra", "cpu", "--extra", "engine", "--group", "dev"]),
        ("cuda", ["--extra", "cu129", "--extra", "engine", "--group", "dev"]),
        ("cu129", ["--extra", "cu129", "--extra", "engine", "--group", "dev"]),
    ],
)
def test_sync_dependencies_preserves_profile_arguments(
    pytestconfig: pytest.Config, fake_uv: Path, profile: str, expected: list[str]
) -> None:
    args = _sync_dependencies(Path(pytestconfig.rootpath), fake_uv, profile, python_version="3.12")

    assert args == ["sync", "--frozen", "--python", "3.12", *expected]


def test_sync_dependencies_accepts_explicit_python_override(pytestconfig: pytest.Config, fake_uv: Path) -> None:
    args = _sync_dependencies(
        Path(pytestconfig.rootpath),
        fake_uv,
        "dev",
        python_version="3.13",
        python_override="/lustre/python3.12",
    )

    assert args == ["sync", "--frozen", "--python", "/lustre/python3.12", "--group", "dev"]


def test_sync_dependencies_propagates_python_resolution_failure(
    pytestconfig: pytest.Config, fake_uv: Path, tmp_path: Path
) -> None:
    repo_root = Path(pytestconfig.rootpath)
    env = os.environ | {"MISE_CONFIG_ROOT": str(tmp_path), "NSS_UV_BIN": str(fake_uv)}
    env.pop("PYTHON_VERSION", None)

    result = subprocess.run(
        ["bash", "-c", 'source "$1"; sync_nss_dependencies dev', "bash", str(repo_root / ".mise/tasks/_lib.sh")],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "PYTHON_VERSION is unset" in result.stderr
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

    assert tasks["check:lock"]["run"] == [
        "uv run --offline --frozen tools/gen_cuda_deps.py cuda_deps.toml --pyproject pyproject.toml --check",
        "uv lock --check",
    ]
    assert tasks["lock:update"]["run"] == "uv lock"


def test_lock_hook_uses_read_only_task_for_both_inputs(pytestconfig: pytest.Config) -> None:
    config = (Path(pytestconfig.rootpath) / ".pre-commit-config.yaml").read_text()
    hook = config.split("      - id: uv-lock", maxsplit=1)[1].split("      - id:", maxsplit=1)[0]

    assert "entry: mise run check:lock" in hook
    assert "pyproject\\.toml" in hook
    assert "cuda_deps\\.toml" in hook


def test_local_gate_composes_check_and_test(pytestconfig: pytest.Config) -> None:
    result = _run_mise(Path(pytestconfig.rootpath), "run", "--dry-run", "check", ":::", "test")

    assert result.returncode == 0, result.stderr

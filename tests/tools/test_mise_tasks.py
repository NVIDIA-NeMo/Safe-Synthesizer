# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


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


@pytest.mark.parametrize("python_version", ["3.11", "3.12", "3.14"])
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

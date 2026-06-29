# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "tools" / "secrets-detector-changed-files.sh"
BASELINE = ".github/workflows/config/.secrets.baseline"


def _run(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, check=False, capture_output=True, text=True)


def _git(repo: Path, *args: str) -> None:
    result = _run("git", *args, cwd=repo)
    assert result.returncode == 0, result.stderr


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    baseline = repo / BASELINE
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}\n")
    (repo / "tracked.txt").write_text("initial\n")
    _commit(repo, "initial")
    return repo


def _detector_env(tmp_path: Path, *, exit_code: int = 0) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "detector-args"
    detector = bin_dir / "detect-secrets-hook"
    detector.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "$DETECTOR_ARGS_FILE"\nexit "$DETECTOR_EXIT_CODE"\n'
    )
    detector.chmod(0o755)
    env = os.environ | {
        "DETECTOR_ARGS_FILE": str(args_file),
        "DETECTOR_EXIT_CODE": str(exit_code),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    return env, args_file


def test_scans_changed_files_and_excludes_baseline(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    changed = repo / "notes" / "file with spaces.txt"
    changed.parent.mkdir()
    changed.write_text("changed\n")
    (repo / BASELINE).write_text('{"results": {}}\n')
    _commit(repo, "change files")
    env, args_file = _detector_env(tmp_path)

    result = _run("bash", str(SCRIPT), "HEAD~1", cwd=repo, env=env)

    assert result.returncode == 0, result.stderr
    args = args_file.read_bytes().rstrip(b"\0").split(b"\0")
    assert args == [b"--baseline", BASELINE.encode(), b"notes/file with spaces.txt"]


def test_succeeds_without_invoking_detector_when_only_baseline_changed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / BASELINE).write_text('{"results": {}}\n')
    _commit(repo, "change baseline")
    env, args_file = _detector_env(tmp_path)

    result = _run("bash", str(SCRIPT), "HEAD~1", cwd=repo, env=env)

    assert result.returncode == 0, result.stderr
    assert "No files to scan after exclusions." in result.stdout
    assert not args_file.exists()


def test_propagates_detector_failure(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "tracked.txt").write_text("changed\n")
    _commit(repo, "change tracked file")
    env, _ = _detector_env(tmp_path, exit_code=7)

    result = _run("bash", str(SCRIPT), "HEAD~1", cwd=repo, env=env)

    assert result.returncode == 7
    assert "SECRET DETECTOR FAILED" in result.stdout

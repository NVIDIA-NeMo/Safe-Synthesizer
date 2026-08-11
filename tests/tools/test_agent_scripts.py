# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SIGNING_HOOK = REPO_ROOT / ".agents" / "hooks" / "enforce-signoff.sh"
WORKTREE_SETUP = REPO_ROOT / ".agents" / "skills" / "git-worktrees" / "scripts" / "setup-worktree.sh"
CURSOR_HOOKS = REPO_ROOT / ".cursor" / "hooks.json"
CLAUDE_SETTINGS = REPO_ROOT / ".claude" / "settings.json"
CODEX_HOOKS = REPO_ROOT / ".codex" / "hooks.json"


def _run_signing_hook(command: str, *, client: str = "cursor") -> subprocess.CompletedProcess[str]:
    if client == "cursor":
        payload = {"command": command}
    elif client == "claude":
        payload = {"tool_input": {"command": command}}
    elif client == "codex":
        payload = {
            "session_id": "test-session",
            "cwd": str(REPO_ROOT),
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": command},
        }
    else:
        raise ValueError(f"Unsupported hook client: {client}")
    return subprocess.run(
        ["bash", str(SIGNING_HOOK)],
        check=False,
        capture_output=True,
        input=json.dumps(payload),
        text=True,
    )


@pytest.fixture
def recording_uv(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_file = tmp_path / "uv-environment"
    uv = bin_dir / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
printf 'args=%s\\n' "$*" > "$CAPTURE_FILE"
	for name in UV_PROJECT_ENVIRONMENT UV_NO_SYNC UV_PROJECT UV_WORKING_DIR PYTHONPATH VIRTUAL_ENV; do
	    printf '%s=%s\\n' "$name" "${!name-<unset>}" >> "$CAPTURE_FILE"
	done
"""
    )
    uv.chmod(0o755)
    return bin_dir, capture_file


@pytest.mark.parametrize("client", ["cursor", "claude", "codex"])
@pytest.mark.parametrize(
    "command",
    [
        'git commit -s -S -m "message"',
        'git commit --signoff --gpg-sign=ABC123 -m "message"',
        'git commit -s -SABC123 -m "message"',
        'FOO=bar env BAZ=qux git --no-pager commit -s -S -m "message"',
        'echo ready && git commit -s -S -m "message"',
        'echo ready; git commit -s -S -m "message"',
        'false || git commit -s -S -m "message"',
        'echo ready | git commit -s -S -m "message"',
        'git -c user.name=Agent commit -s -S -m "message"',
        'git -C /tmp commit -s -S -m "message"',
        'git commit -m "first; second" -s -S',
        'git commit -m "say \\"hello; world\\"" -s -S',
        'git commit -s -S -m "--no-signoff; --no-gpg-sign"',
    ],
)
def test_signing_hook_allows_supported_commit_forms(command: str, client: str) -> None:
    result = _run_signing_hook(command, client=client)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("client", ["cursor", "claude", "codex"])
@pytest.mark.parametrize(
    ("command", "expected_error"),
    [
        ('git commit -m "message"', "DCO sign-off"),
        ('git commit -S -m "message"', "DCO sign-off"),
        ('git commit -s -m "message"', "GPG-signed"),
        ('git commit -s -S --no-signoff -m "message"', "DCO sign-off"),
        ('git commit -s -S --no-gpg-sign -m "message"', "GPG-signed"),
        ('git commit --no-signoff -s -S -m "message"', "DCO sign-off"),
        ('git commit --no-gpg-sign -s -S -m "message"', "GPG-signed"),
        ('git commit -s -S "--no-signoff" -m "message"', "Quoted non-message"),
        ('git commit -s -S --no-sign"off" -m "message"', "Quoted non-message"),
        ('git commit -s -S "--no-gpg-sign" -m "message"', "Quoted non-message"),
        ("git commit -mSIGNEDsignoff", "DCO sign-off"),
        ('FOO=bar env BAZ=qux git commit -S -m "message"', "DCO sign-off"),
        ('git --no-pager commit -S -m "message"', "DCO sign-off"),
        ('echo ready && git commit -S -m "message"', "DCO sign-off"),
        ('(git commit -S -m "message")', "DCO sign-off"),
    ],
)
def test_signing_hook_blocks_unsupported_commit_forms(command: str, expected_error: str, client: str) -> None:
    result = _run_signing_hook(command, client=client)

    assert result.returncode == 2, result.stderr
    assert expected_error in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        'bash -c "git commit -m bad"',
        'echo "text; git commit -S"',
    ],
)
def test_signing_hook_ignores_commands_outside_its_supported_scope(command: str) -> None:
    result = _run_signing_hook(command)

    assert result.returncode == 0, result.stderr


def test_codex_registers_signing_hook() -> None:
    config = json.loads(CODEX_HOOKS.read_text())

    assert config["hooks"]["PreToolUse"] == [
        {
            "matcher": "^Bash$",
            "hooks": [
                {
                    "type": "command",
                    "command": 'bash "$(git rev-parse --show-toplevel)/.agents/hooks/enforce-signoff.sh"',
                    "timeout": 5,
                    "statusMessage": "Checking commit signing requirements",
                }
            ],
        }
    ]


def test_cursor_registers_signing_hook() -> None:
    config = json.loads(CURSOR_HOOKS.read_text())

    assert config == {
        "version": 1,
        "hooks": {
            "beforeShellExecution": [
                {"command": ".cursor/hooks/enforce-signoff.sh"},
            ]
        },
    }


def test_claude_registers_signing_hook() -> None:
    config = json.loads(CLAUDE_SETTINGS.read_text())

    assert config == {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": '"$CLAUDE_PROJECT_DIR"/.claude/hooks/enforce-signoff.sh',
                        }
                    ],
                }
            ]
        }
    }


@pytest.mark.parametrize(
    ("adapter", "target"),
    [
        (REPO_ROOT / ".cursor" / "hooks" / "enforce-signoff.sh", "../../.agents/hooks/enforce-signoff.sh"),
        (REPO_ROOT / ".claude" / "hooks" / "enforce-signoff.sh", "../../.agents/hooks/enforce-signoff.sh"),
    ],
)
def test_client_signing_hook_adapters_are_relative_symlinks(adapter: Path, target: str) -> None:
    assert adapter.is_symlink()
    assert os.readlink(adapter) == target
    assert adapter.resolve() == SIGNING_HOOK.resolve()


def test_worktree_setup_clears_inherited_environment(
    recording_uv: tuple[Path, Path],
) -> None:
    bin_dir, capture_file = recording_uv
    env = os.environ | {
        "CAPTURE_FILE": str(capture_file),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHONPATH": "/shared/src",
        "UV_NO_SYNC": "1",
        "UV_PROJECT": "/shared/project",
        "UV_PROJECT_ENVIRONMENT": "/shared/.venv",
        "UV_WORKING_DIR": "/shared/worktree",
        "VIRTUAL_ENV": "/shared/.venv",
    }

    result = subprocess.run(
        ["bash", str(WORKTREE_SETUP)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert capture_file.read_text().splitlines() == [
        "args=sync --frozen",
        "UV_PROJECT_ENVIRONMENT=<unset>",
        "UV_NO_SYNC=<unset>",
        "UV_PROJECT=<unset>",
        "UV_WORKING_DIR=<unset>",
        "PYTHONPATH=<unset>",
        "VIRTUAL_ENV=<unset>",
    ]

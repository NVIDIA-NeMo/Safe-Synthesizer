#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SessionStart hook for both Cursor and Claude Code.
# Reports venv state and sets one up if absent.
#
# Cursor worktrees set $ROOT_WORKTREE_PATH; normal sessions and Claude Code do not.
# Output format differs by platform:
#   Cursor:     {"additional_context": "..."} -- injected into agent system context
#   Claude Code: plain stdout -- added as context on SessionStart (exit 0)

set -eu

# Ensure tools installed in ~/.local/bin (gh, glab, etc.) are on PATH.
export PATH="$HOME/.local/bin:$PATH"

IN_CURSOR_WORKTREE="${ROOT_WORKTREE_PATH:+yes}"
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

if [ -d ".venv/bin" ]; then
    VENV_STATUS="local .venv present"
    VENV_HINT="To recreate after changing deps: uv sync (without --frozen)."
else
    echo "No .venv found -- running uv sync --frozen..." >&2
    uv sync --frozen >&2
    VENV_STATUS="local .venv created by session hook (uv sync --frozen)"
    VENV_HINT="To recreate after changing deps: uv sync (without --frozen)."
fi

if [ -n "$IN_CURSOR_WORKTREE" ]; then
    jq -n \
        --arg status "$VENV_STATUS" \
        --arg hint "$VENV_HINT" \
        --arg branch "$BRANCH" \
        --arg root "${ROOT_WORKTREE_PATH:-}" \
        '{additional_context: "WORKTREE SESSION\nBranch: \($branch)\nMain worktree: \($root)\nVenv: \($status)\n\($hint)"}'
else
    echo "Branch: $BRANCH | Venv: $VENV_STATUS | $VENV_HINT"
fi

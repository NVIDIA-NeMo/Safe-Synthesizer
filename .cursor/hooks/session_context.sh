#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SessionStart hook for both Cursor and Claude Code.
# Reports venv state and sets one up if absent.
#
# Platform detection:
#   Claude Code sets $CLAUDE_PROJECT_DIR  -> plain stdout (exit 0)
#   Cursor worktrees set $ROOT_WORKTREE_PATH -> {"additional_context": "..."}
#   Cursor normal sessions              -> {"additional_context": "..."}

set -eu

# Ensure tools installed in ~/.local/bin (gh, glab, etc.) are on PATH.
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"

# Auto-trusting repo-local mise config is security-sensitive; only do it when
# the caller opts in (or set MISE_TRUSTED_CONFIG_PATHS in your shell profile).
if command -v mise >/dev/null 2>&1 && [ "${CURSOR_ALLOW_MISE_TRUST:-0}" = "1" ]; then
    mise trust --quiet 2>/dev/null || true
fi

# Source project-local env if present (API keys, secrets, etc.).
# mise.local.toml, .env, and .env.local are loaded automatically by mise;
# source them here as a fallback for hook contexts.
for _envfile in .env .env.local .local.envrc; do
    if [ -f "$_envfile" ]; then
        # shellcheck disable=SC1090
        set -a && . "./$_envfile" && set +a
    fi
done
unset _envfile

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

if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    # Claude Code: plain stdout becomes session context.
    echo "Branch: $BRANCH | Venv: $VENV_STATUS | $VENV_HINT"
elif [ -n "${ROOT_WORKTREE_PATH:+yes}" ]; then
    # Cursor worktree session.
    jq -n \
        --arg status "$VENV_STATUS" \
        --arg hint "$VENV_HINT" \
        --arg branch "$BRANCH" \
        --arg root "${ROOT_WORKTREE_PATH:-}" \
        '{additional_context: "WORKTREE SESSION\nBranch: \($branch)\nMain worktree: \($root)\nVenv: \($status)\n\($hint)"}'
else
    # Cursor normal session.
    jq -n \
        --arg status "$VENV_STATUS" \
        --arg hint "$VENV_HINT" \
        --arg branch "$BRANCH" \
        '{additional_context: "Branch: \($branch)\nVenv: \($status)\n\($hint)"}'
fi

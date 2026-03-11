#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cursor parallel-agent worktree setup script.
# $ROOT_WORKTREE_PATH is set by Cursor to the main worktree path.

set -eu

MAIN_HEAD=$(git -C "$ROOT_WORKTREE_PATH" rev-parse HEAD)
if [ -n "$(git diff "$MAIN_HEAD" -- uv.lock)" ]; then
    echo "Note: uv.lock differs from main worktree HEAD -- worktree has diverged dependencies"
else
    echo "uv.lock matches main worktree HEAD"
fi

# Bare --frozen installs the base environment. For GPU dev work (ty, import
# checks, GPU tests) run the full command manually after setup:
#   uv sync --frozen --extra cu128 --extra engine --group dev
uv sync --frozen
echo "Venv ready: $(pwd)/.venv"
echo "Note: for GPU extras run: uv sync --frozen --extra cu128 --extra engine --group dev"

if [ -f "$ROOT_WORKTREE_PATH/.local.envrc" ]; then
    cp "$ROOT_WORKTREE_PATH/.local.envrc" .local.envrc
    echo "Copied .local.envrc from main worktree"
fi

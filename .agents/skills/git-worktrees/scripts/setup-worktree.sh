#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare a Safe-Synthesizer worktree with its base Python environment.

set -eu

MAIN_CHECKOUT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
MAIN_HEAD=$(git -C "$MAIN_CHECKOUT" rev-parse HEAD)

if [ -n "$(git diff "$MAIN_HEAD" -- uv.lock)" ]; then
    echo "Note: uv.lock differs from main worktree HEAD; dependencies have diverged"
else
    echo "uv.lock matches main worktree HEAD"
fi

uv sync --frozen
echo "Base environment ready: $(pwd)/.venv"
echo "For a complete profile run: mise run setup && mise run bootstrap-nss cpu (or cu129)"

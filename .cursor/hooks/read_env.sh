#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# read_env.sh - preToolUse hook that sets up PATH and loads local env vars

# Ensure mise-managed tools are on PATH.
# Auto-trusting repo-local config is security-sensitive; only do it when
# the caller opts in (or configure trust via shell-profile env vars).
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
if command -v mise >/dev/null 2>&1 && [ "${CURSOR_ALLOW_MISE_TRUST:-0}" = "1" ]; then
    mise trust --quiet 2>/dev/null
fi

# Load project-local env vars (if the file exists and auto-trust is opted in).
# Without CURSOR_ALLOW_DIRENV_TRUST, the file is still sourced directly but
# direnv won't be told to trust/execute the repo's .envrc.
if [ -f .local.envrc ]; then
    if command -v direnv >/dev/null 2>&1 && [ "${CURSOR_ALLOW_DIRENV_TRUST:-0}" = "1" ]; then
        direnv allow
    fi
    source .local.envrc
fi

# Exit successfully
exit 0

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

# Load project-local env vars. mise.local.toml, .env, and .env.local are
# handled automatically by mise; source them here as a fallback for contexts
# where mise is not fully activated.
for _envfile in .env .env.local .local.envrc; do
    # shellcheck disable=SC1090
    [ -f "$_envfile" ] && . "./$_envfile"
done
unset _envfile

# Exit successfully
exit 0

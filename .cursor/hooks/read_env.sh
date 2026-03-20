#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# read_env.sh - preToolUse hook that sets up PATH and loads local env vars

# Ensure mise-managed tools are on PATH and config is trusted
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
if command -v mise >/dev/null 2>&1; then
    mise trust --quiet 2>/dev/null
fi

# Read the environment variables from the local env file (if it exists)
if [ -f .local.envrc ]; then
    if command -v direnv >/dev/null 2>&1; then
        direnv allow
    fi
    source .local.envrc
fi

# Exit successfully
exit 0

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# read_env.sh - preToolUse hook that sets up PATH and loads local env vars

# Ensure mise-managed tools are on PATH
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"

# Read the environment variables from the local env file (if it exists)
if [ -f .local.envrc ]; then
    direnv allow
    source .local.envrc
fi

# Exit successfully
exit 0

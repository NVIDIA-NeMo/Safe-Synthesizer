#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Blocks `git commit` without --signoff/-s or without --gpg-sign/-S.
# Exit code 2 = deny action on both Cursor and Claude Code.
# Stderr is fed back to the agent as the deny reason on both platforms.

set -eu

json_input=$(cat)

# Cursor puts the command at .command (beforeShellExecution);
# Claude Code puts it at .tool_input.command (PreToolUse for Bash).
command=$(echo "$json_input" | jq -r '.command // .tool_input.command // empty')

if echo "$command" | grep -qE '^git commit'; then
    if ! echo "$command" | grep -qE '(--signoff| -[a-zA-Z]*s)'; then
        echo "All commits require DCO sign-off. Re-run with --signoff (or -s)." >&2
        exit 2
    fi
    if ! echo "$command" | grep -qE '(--gpg-sign| -[a-zA-Z]*S)'; then
        echo "All commits must be GPG-signed. Re-run with --gpg-sign (or -S)." >&2
        exit 2
    fi
fi

exit 0

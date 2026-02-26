#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Blocks `git commit` without --signoff/-s. Cursor exit code 2 = deny action.
# The agentMessage tells the agent to retry with --signoff.

json_input=$(cat)
command=$(echo "$json_input" | jq -r '.command // empty')

if echo "$command" | grep -qE '^git commit' && ! echo "$command" | grep -qE '(--signoff| -[a-zA-Z]*s)'; then
    echo '{"exitCode": 2, "agentMessage": "All commits require DCO sign-off. Re-run with --signoff (or -s)."}'
    exit 2
fi

exit 0

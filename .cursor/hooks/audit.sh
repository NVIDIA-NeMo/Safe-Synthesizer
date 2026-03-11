#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

json_input=$(cat)
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
event=$(echo "$json_input" | jq -r '.hook_event_name // "unknown"')

CURSOR_AUDIT_LOG="${CURSOR_AUDIT_LOG:-"${HOME}/.cursor/audit.log"}"
mkdir -p "$(dirname "$CURSOR_AUDIT_LOG")"
echo "[$timestamp] [$event] $json_input" >> "$CURSOR_AUDIT_LOG"

exit 0

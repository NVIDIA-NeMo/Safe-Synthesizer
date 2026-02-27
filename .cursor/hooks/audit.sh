#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

# audit.sh - Hook script that writes all JSON input to /tmp/agent-audit.log
# This script is designed to be called by Cursor's hooks system for auditing purposes

# Read JSON input from stdin
json_input=$(cat)

# Create timestamp for the log entry
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

LOG_FILE="/tmp/cursor-agent-audit-log/${timestamp}-agent.json"
# Create the log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Write the timestamped JSON entry to the audit log
echo "[$timestamp] $json_input" >> "$LOG_FILE"

# Exit successfully
exit 0

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run detect-secrets against files changed relative to a target branch.
#
# Usage:
#   uv run --no-project --with detect-secrets==1.5.0 -- \
#     bash tools/secrets-detector-changed-files.sh origin/main

set -euo pipefail

readonly baseline=".github/workflows/config/.secrets.baseline"
target_branch="${1:-origin/${GITHUB_BASE_REF:-main}}"

all_files="$(mktemp)"
scan_files="$(mktemp)"
trap 'rm -f "$all_files" "$scan_files"' EXIT

git diff --name-only --diff-filter=d --merge-base "$target_branch" -z > "$all_files"
grep -z -v -F -x -- "$baseline" "$all_files" > "$scan_files" || true

echo "Target branch: ${target_branch}"
if [ ! -s "$scan_files" ]; then
    echo "No files to scan after exclusions."
    exit 0
fi

echo "Files to scan:"
tr '\0' '\n' < "$scan_files"

if ! command -v detect-secrets-hook > /dev/null; then
    echo "detect-secrets-hook is unavailable. Run this tool through the uv command in its usage block." >&2
    exit 125
fi

set +e
scan_paths=()
mapfile -d '' scan_paths < "$scan_files"
detect-secrets-hook --baseline "$baseline" "${scan_paths[@]}"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    exit 0
fi

echo ""
echo "========================================"
echo "        SECRET DETECTOR FAILED"
echo "========================================"
echo ""
echo "The secret detector found potential secrets in your changes."
echo ""
echo "If this is a real secret:"
echo "  1. Remove the secret from the branch."
echo "  2. Rotate or revoke the exposed credential."
echo "  3. Remove or squash the commit containing the secret."
echo ""
echo "If this is a false positive, update and review ${baseline}."
echo "See https://github.com/Yelp/detect-secrets for baseline management."
exit "$exit_code"

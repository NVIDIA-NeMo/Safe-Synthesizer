#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Run detect-secrets against files changed relative to the target branch.
#
# Usage:
#   bash tools/secrets-detector-changed-files.sh
#   bash tools/secrets-detector-changed-files.sh origin/main
#
# The caller must provide detect-secrets-hook on PATH. CI installs the pinned
# version before invoking this script; local runs can use:
#
#   mise exec -- uv run --no-project --with detect-secrets==1.5.0 \
#     bash tools/secrets-detector-changed-files.sh origin/main
#

set -euo pipefail

target_branch="${1:-origin/${GITHUB_BASE_REF:-main}}"
echo "Target branch is: ${target_branch}"

all_files="$(mktemp)"
scan_files="$(mktemp)"
trap 'rm -f "$all_files" "$scan_files"' EXIT

git diff --name-only --diff-filter=d --merge-base "$target_branch" -z > "$all_files"
grep -z -v -E '^\.github/workflows/config/\.secrets\.baseline$' "$all_files" > "$scan_files" || true

echo "Detecting secrets in the following files:"
if [ ! -s "$scan_files" ]; then
    echo "No files to scan after exclusions."
    exit 0
fi
tr '\0' '\n' < "$scan_files"

set +e
xargs -0 -r detect-secrets-hook --baseline .github/workflows/config/.secrets.baseline < "$scan_files"
exit_code=$?
set -e

if [ "$exit_code" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "        SECRET DETECTOR FAILED"
    echo "========================================"
    echo ""
    echo "The secret detector found potential secrets in your changes."
    echo ""
    echo "HOW TO FIX:"
    echo ""
    echo "If this is a real secret:"
    echo "  1. Remove the secret from your code immediately"
    echo "  2. Rotate or revoke the exposed credential"
    echo "  3. Remove or squash the commit containing the secret on your branch"
    echo ""
    echo "If this is a false positive:"
    echo "  1. Run: detect-secrets scan --exclude-files 'pyproject\\.toml|\\.github/workflows/config/\\.secrets\\.baseline' --disable-plugin KeywordDetector > .github/workflows/config/.secrets.baseline"
    echo "  2. Review the updated baseline file to ensure the added entries are real false positives"
    echo "  3. Commit the updated baseline file"
    echo ""
    echo "For more information, see: https://github.com/Yelp/detect-secrets"
    echo ""
    exit "$exit_code"
fi

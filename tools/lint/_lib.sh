#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# _lib.sh -- shared helpers for lint/format/typecheck scripts
#
# Source this file; do not execute it directly.
#
# Provides:
#   REPO_ROOT                    -- absolute path to the repo root
#   require_tool <name>          -- die if <name> is not on PATH
#   collect_py_files "$@"        -- emit file list on stdout
#
# collect_py_files recognises --check and exports CHECK_MODE=true/false
# so callers (format.sh) can inspect it.
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# require_tool
# ---------------------------------------------------------------------------
require_tool() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "ERROR: $name not found on PATH." >&2
        echo "       Run: make bootstrap-tools && export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# collect_py_files -- parse args and emit a file list on stdout
#
# Modes:
#   (no args)                    → all tracked .py files
#   file1.py file2.py ...        → those exact files
#
# Also strips --check from the arg list and exports CHECK_MODE.
# ---------------------------------------------------------------------------
CHECK_MODE=false
export CHECK_MODE

collect_py_files() {
    local -a rest=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check)
                CHECK_MODE=true
                shift
                ;;
            *)
                rest+=("$1")
                shift
                ;;
        esac
    done

    export CHECK_MODE

    local files=""

    if [[ ${#rest[@]} -gt 0 ]]; then
        files=$(printf '%s\n' "${rest[@]}")
    else
        files=$(git ls-files '*.py')
    fi

    if [[ -z "$files" ]]; then
        echo "No Python files to check" >&2
        exit 0
    fi

    echo "$files"
}

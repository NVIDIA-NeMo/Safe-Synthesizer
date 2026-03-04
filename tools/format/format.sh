#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# format.sh -- format (or check formatting of) Python files with ruff
#
# Usage:
#   ./format.sh                          # fix mode, all tracked .py files
#   ./format.sh --check                  # check mode (exit 1 if unformatted)
#   ./format.sh src/foo.py bar.py        # fix specific files
#   ./format.sh --check src/foo.py       # check mode on specific files
#
# Copyright headers are handled separately by copyright_fixer.py
# (called from `make format`, `make lint`, and the pre-commit copyright-fix hook).
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=../lint/_lib.sh
source "$REPO_ROOT/tools/lint/_lib.sh"

require_tool ruff

files=$(collect_py_files "$@")

if [[ "$CHECK_MODE" == true ]]; then
    # shellcheck disable=SC2086
    ruff format --check $files
else
    # shellcheck disable=SC2086
    ruff format $files
    # shellcheck disable=SC2086
    ruff check --fix $files
fi

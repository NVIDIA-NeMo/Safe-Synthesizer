#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# ruff-lint.sh -- lint Python files with ruff
#
# Usage:
#   ./ruff-lint.sh                     # all tracked .py files
#   ./ruff-lint.sh src/foo.py bar.py   # specific files
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=_lib.sh
source "$REPO_ROOT/tools/lint/_lib.sh"

require_tool ruff

files=$(collect_py_files "$@")

# shellcheck disable=SC2086
ruff check $files

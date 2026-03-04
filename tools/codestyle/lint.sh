#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# lint.sh -- lint Python files with ruff
#
# Usage:
#   ./lint.sh                            # all tracked .py files
#   ./lint.sh src/foo.py bar.py          # specific files
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=_lib.sh
source "$REPO_ROOT/tools/codestyle/_lib.sh"

require_tool ruff

collect_py_files "$@"
[[ ${#PY_FILES[@]} -eq 0 ]] && exit 0

ruff check "${PY_FILES[@]}"

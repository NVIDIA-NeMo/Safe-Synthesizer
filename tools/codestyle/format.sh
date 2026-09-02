#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# format.sh -- format (or check formatting of) Python files with ruff
#
# Usage:
#   ./format.sh                          # fix mode: ruff format + ruff check --fix
#   ./format.sh --check                  # check mode: ruff format --check (exit 1 if unformatted)
#   ./format.sh src/foo.py bar.py        # fix specific files
#   ./format.sh --check src/foo.py       # check mode on specific files
#
# Lint-rule violations (ruff check without --fix) are handled by ruff_check.sh.
# Type-checking is handled by typecheck.sh.
# Copyright headers are handled separately by copyright_fixer.py.
#
# Note: `ty check --fix` (ty 0.0.32+) is intentionally not wired in here yet.
# The autofix surface is still small and the CLI semantics (file discovery vs.
# explicit paths, rule suppression, exit codes) are evolving. When we do wire
# it, it goes in the fix branch below; see the commented placeholder.
#

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=_lib.sh
source "$REPO_ROOT/tools/codestyle/_lib.sh"

collect_py_files "$@"
[[ ${#PY_FILES[@]} -eq 0 ]] && exit 0

if [[ "$CHECK_MODE" == true ]]; then
    ruff format --check "${PY_FILES[@]}"
else
    ruff format "${PY_FILES[@]}"
    # this does import sorting and autofixes
    ruff check --fix "${PY_FILES[@]}"
    # Future: ty autofix goes here once the --fix surface stabilizes. Shape:
    #   ty check --fix --exit-zero --force-exclude "${PY_FILES[@]}"
    # --force-exclude so [tool.ty.src.exclude] in pyproject.toml is honored
    # when paths are passed explicitly; --exit-zero so non-fixable diagnostics
    # don't fail `mise run format` (`mise run check:type` is the gate). When enabling,
    # also add "./.agents/" to [tool.ty.src.exclude] -- PEP 723 uv scripts
    # there would otherwise show up when PY_FILES is passed on the CLI.
fi

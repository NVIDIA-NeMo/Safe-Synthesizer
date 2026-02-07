#!/usr/bin/env bash

#
# format.sh - Format (or check formatting of) changed Python files
#
# Usage:
#   ./format.sh [MERGE_BASE_SHA]          # fix mode (default) — rewrites files
#   ./format.sh --check [MERGE_BASE_SHA]  # check mode — exits 1 if unformatted
#

set -euo pipefail

NSS_ROOT_PATH="${NSS_ROOT_PATH:-$(git rev-parse --show-toplevel)}"

# Parse --check flag
CHECK_MODE=false
MERGE_BASE_SHA=""
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_MODE=true ;;
        *)       MERGE_BASE_SHA="$arg" ;;
    esac
done

if [ -z "$MERGE_BASE_SHA" ]; then
    if git branch -l | grep "main" > /dev/null; then
        MERGE_BASE_SHA="main"
    else
        echo "Merge Base SHA is required"
        exit 1
    fi
fi

# Get the list of changed Python files
files=$(git diff "$MERGE_BASE_SHA" --cached --name-only --diff-filter=ACMR | grep '\.py$' || true)

if [ -z "$files" ]; then
	echo "No Python files to check"
	exit 0
fi

# Filter out files excluded in pyproject.toml's [tool.ty.src].exclude
filtered_files=$(echo "$files" | uv run --frozen python tools/lint/filter_ty_exclusions.py)

if [ -z "$filtered_files" ]; then
	echo "No Python files to check (all files are excluded)"
	exit 0
fi

# Resolve ruff
if ! which ruff > /dev/null; then
    echo "ruff not found, falling back to uvx"
    RUFF="uvx ruff"
else
    RUFF="ruff"
fi

if [ "$CHECK_MODE" = true ]; then
    echo "Running in check mode (no files will be modified)"
    # shellcheck disable=SC2086
    $RUFF format --check "$NSS_ROOT_PATH" $filtered_files
    # shellcheck disable=SC2086
    $RUFF check --select I "$NSS_ROOT_PATH" $filtered_files
    # shellcheck disable=SC2086
    uv run --script tools/lint/copyright_fixer.py --check $filtered_files
else
    # shellcheck disable=SC2086
    $RUFF format "$NSS_ROOT_PATH" $filtered_files
    # shellcheck disable=SC2086
    $RUFF check --select I --fix "$NSS_ROOT_PATH" $filtered_files
    # shellcheck disable=SC2086
    uv run --script tools/lint/copyright_fixer.py $filtered_files
fi

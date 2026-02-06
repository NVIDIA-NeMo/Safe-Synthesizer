#!/usr/bin/env bash
#
# format.sh - Format the code for the whole repo, not just the changed files
#
# Usage: ./format.sh
#

set -euo pipefail

NSS_ROOT_PATH="${NSS_ROOT_PATH:-$(git rev-parse --show-toplevel)}"

MERGE_BASE_SHA="${1:-}"

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

# Run ruff check on the filtered files
if ! which ruff > /dev/null; then
    echo "ruff not found"
    RUFF="uvx ruff"
else
    RUFF="ruff"
fi

# shellcheck disable=SC2086
$RUFF format "$NSS_ROOT_PATH" $filtered_files && $RUFF check --select I --fix "$NSS_ROOT_PATH" $filtered_files # no quotes around $filtered_files to preserve newlines

# shellcheck disable=SC2086
uv run python tools/lint/copyright_fixer.py $filtered_files
#!/bin/bash
set -e

# Get the list of changed Python files
MERGE_BASE_SHA="${1:-}"

if [ -z "$MERGE_BASE_SHA" ]; then
    if git branch -l | grep "main" > /dev/null; then
        MERGE_BASE_SHA="main"
    else
        echo "Merge Base SHA is required"
        exit 1
    fi
fi
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

# shellcheck disable=SC2086
ty check $filtered_files # no quotes around $filtered_files to preserve newlines

#!/bin/bash
set -e

# Get the list of changed Python files
files=$(git diff --cached --name-only --diff-filter=ACMR | grep '\.py$' || true)

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

# Run ty check on the filtered files
uv run ty check $filtered_files

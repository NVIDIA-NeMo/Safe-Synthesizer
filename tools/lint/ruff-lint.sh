#!/usr/bin/env bash
#
# ruff-lint.sh - Lint the code with ruff
#
# Usage: ./ruff-lint.sh
#

set -euo pipefail

NSS_ROOT_PATH="${NSS_ROOT_PATH:-$(git rev-parse --show-toplevel)}"

uv run --frozen ruff check --fix "$(NSS_ROOT_PATH)" 
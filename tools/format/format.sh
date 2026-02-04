#!/usr/bin/env bash
#
# format.sh - Format the code for the whole repo, not just the changed files
#
# Usage: ./format.sh
#

set -euo pipefail

NSS_ROOT_PATH="${NSS_ROOT_PATH:-$(git rev-parse --show-toplevel)}"

uv run --frozen ruff format "$(NSS_ROOT_PATH)"  && uv run --frozen ruff check --select I --fix "$(NSS_ROOT_PATH)"
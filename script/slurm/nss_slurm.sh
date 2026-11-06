#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run nss_slurm.py on the login node with the settings from env_variables.sh.
#
# Usage: bash script/slurm/nss_slurm.sh {submit|build-image|download-models} [options]
# submit_slurm_jobs.sh, build_image.sh, and download_models.sh call this for you.
#
# uv's cache and any Python that uv downloads go to login-node local disk
# (/tmp), not to /home or Lustre.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

if [[ -z "${USER_NAME:-}" ]]; then
  echo "ERROR: USER_NAME is not set. Export your cluster user name first, e.g. export USER_NAME=jdoe" >&2
  exit 1
fi
# shellcheck source=script/slurm/env_variables.sh
source "${SCRIPT_DIR}/env_variables.sh"

# CONFIGS is a bash array, which cannot be exported; pass it as a comma-separated list.
NSS_DEFAULT_CONFIGS="$(IFS=,; echo "${CONFIGS[*]}")"
export NSS_DEFAULT_CONFIGS

login_uv_dir="/tmp/${USER:-$(id -un)}/nss-uv"
export UV_CACHE_DIR="${login_uv_dir}/cache"
export UV_PYTHON_INSTALL_DIR="${login_uv_dir}/python"
unset UV_PYTHON_BIN_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR VIRTUAL_ENV
# Keep the CLI's output in order with sbatch messages when piped.
export PYTHONUNBUFFERED=1

if ! command -v uv > /dev/null 2>&1; then
  echo "ERROR: uv is not on PATH. Install it (for example with mise) on the login node." >&2
  exit 1
fi
# --no-config: the CLI needs no settings from the repo's pyproject (such as its
# required uv version), only the standard library.
exec uv run --quiet --no-config --script "${SCRIPT_DIR}/nss_slurm.py" "$@"

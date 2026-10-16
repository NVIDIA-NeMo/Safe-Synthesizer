#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the team-shared Slurm image for NSS on the cluster, without Docker.
#
# The image holds system packages, uv, a managed Python, and the locked
# dependencies in /opt/venv, so jobs never import code from Lustre. Jobs run
# `uv sync` at start to install the project and any dependency differences
# from their branch. Keep the extras and uv settings in sync with
# containers/Dockerfile.cuda.
#
# Usage (login node, from a checkout in /home):
#   bash script/slurm/build_image.sh [--partition P] [--time T] [--wait] [--force] [--dry-run]
#
# Submitting is handled by nss_slurm.py ("build-image"). This script keeps the
# compute-node steps, which run before the image has Python:
#   --job        batch step on the compute node; runs the container and publishes the image
#   --container  inside the build container; installs packages, uv, Python, and dependencies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
readonly SCRIPT_DIR REPO_ROOT

readonly CUDA_EXTRA="cu129"
readonly SQUASH_OPTIONS="-comp zstd -Xcompression-level 3 -b 1M -exit-on-error"

# Batch step on the compute node: run the build container and publish the image.
run_build_job() {
  : "${NSS_IMAGES_DIR:?}" "${NSS_IMAGE_NAME:?}" "${NSS_BASE_IMAGE:?}" "${USER_NAME:?}"
  # Group-writable, so other team members can remove superseded images.
  umask 002
  local tmp_image="${NSS_IMAGES_DIR}/.tmp-${NSS_IMAGE_NAME}.${SLURM_JOB_ID}"
  local final_image="${NSS_IMAGES_DIR}/${NSS_IMAGE_NAME}"
  # The build reuses the per-node uv cache that jobs use.
  local node_cache="/raid/scratch/${USER_NAME}/nss-cache"
  mkdir -p "${node_cache}/uv"
  # Expand now: tmp_image is local and no longer exists when the EXIT trap runs.
  # shellcheck disable=SC2064
  trap "rm -f '${tmp_image}'" EXIT

  ENROOT_SQUASH_OPTIONS="${SQUASH_OPTIONS}" \
  NSS_UV_CACHE_DIR="${node_cache}/uv" \
    srun --ntasks=1 \
      --container-image="${NSS_BASE_IMAGE}" \
      --no-container-mount-home \
      --container-mounts="${REPO_ROOT}:${REPO_ROOT}:ro,/raid/scratch:/raid/scratch" \
      --container-save="${tmp_image}" \
      bash "${REPO_ROOT}/script/slurm/build_image.sh" --container

  # NSS_FORCE=1 (build-image --force) replaces an existing image; otherwise the
  # first published image wins. Running jobs keep the image they already unpacked.
  if [[ -e "${final_image}" && "${NSS_FORCE:-0}" != "1" ]]; then
    echo "Another build already published ${final_image}; discarding this one."
    exit 0
  fi
  chmod 664 "${tmp_image}"
  mv -f "${tmp_image}" "${final_image}"
  echo "Published ${final_image} ($(du -h "${final_image}" | cut -f1))"
}

# Inside the build container (runs as root mapped to the submitting user).
build_in_container() {
  : "${NSS_UV_URL:?}" "${NSS_UV_SHA256:?}" "${NSS_UV_CACHE_DIR:?}"
  local python_version
  python_version="$(tr -d '[:space:]' < "${REPO_ROOT}/.python-version")"

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends build-essential ca-certificates curl git libgomp1
  rm -rf /var/lib/apt/lists/*

  local uv_tmp
  uv_tmp="$(mktemp -d)"
  curl -fsSL "${NSS_UV_URL}" -o "${uv_tmp}/uv.tar.gz"
  echo "${NSS_UV_SHA256}  ${uv_tmp}/uv.tar.gz" | sha256sum -c -
  tar -xzf "${uv_tmp}/uv.tar.gz" -C "${uv_tmp}" --strip-components=1
  install -m 0755 "${uv_tmp}/uv" "${uv_tmp}/uvx" /usr/local/bin/
  rm -rf "${uv_tmp}"

  export UV_CACHE_DIR="${NSS_UV_CACHE_DIR}"
  export UV_PYTHON_INSTALL_DIR=/opt/python
  export UV_PROJECT_ENVIRONMENT=/opt/venv
  export UV_MANAGED_PYTHON=1
  export UV_COMPILE_BYTECODE=1
  export UV_LINK_MODE=copy
  unset UV_PYTHON_BIN_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR VIRTUAL_ENV

  uv python install --no-bin "${python_version}"
  cd "${REPO_ROOT}"
  uv sync --frozen --no-install-project --no-dev --extra engine --extra "${CUDA_EXTRA}"

  mkdir -p /opt/nss-image
  cp uv.lock .python-version /opt/nss-image/
  {
    echo "built_at=$(date -u +%FT%TZ)"
    echo "built_by=${USER_NAME:-unknown}"
    echo "git_commit=$(git -c safe.directory='*' rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "uv=$(uv --version)"
    echo "python=$(/opt/venv/bin/python --version 2>&1)"
    echo "extras=engine,${CUDA_EXTRA}"
  } > /opt/nss-image/build-info
  cat /opt/nss-image/build-info

  /opt/venv/bin/python -c "import torch; print('torch', torch.__version__)"
}

case "${1:-}" in
  --job) run_build_job ;;
  --container) build_in_container ;;
  *) exec bash "${SCRIPT_DIR}/nss_slurm.sh" build-image "$@" ;;
esac

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
# The script runs in three modes:
#   submit     (default) compute the image name and submit a build job
#   --job      batch step on the compute node; runs the container and publishes the image
#   --container inside the build container; installs packages, uv, Python, and dependencies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
readonly SCRIPT_DIR REPO_ROOT

readonly CUDA_EXTRA="cu129"
readonly SQUASH_OPTIONS="-comp zstd -Xcompression-level 3 -b 1M -exit-on-error"

# Print the image file name for the current checkout:
# nss-slurm-py<python version>-<first 12 hex of sha256(uv.lock)>.sqsh
image_name() {
  local python_version lock_hash
  python_version="$(tr -d '[:space:]' < "${REPO_ROOT}/.python-version")"
  lock_hash="$(sha256sum "${REPO_ROOT}/uv.lock" | cut -c1-12)"
  printf 'nss-slurm-py%s-%s.sqsh\n' "${python_version}" "${lock_hash}"
}

# Print "<url> <sha256>" for the linux-x64 uv release pinned in mise.lock.
pinned_uv_release() {
  awk '
    /^\[tools\.uv\."platforms\.linux-x64"\]/ { in_section = 1; next }
    /^\[/ { in_section = 0 }
    in_section && $1 == "checksum" { gsub(/"|sha256:/, "", $3); checksum = $3 }
    in_section && $1 == "url" { gsub(/"/, "", $3); url = $3 }
    END { if (url && checksum) print url, checksum; else exit 1 }
  ' "${REPO_ROOT}/mise.lock"
}

submit_build() {
  local partition="cpu_short" time_limit="01:00:00" wait_flag="" force="" dry_run=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --partition) partition="${2:?--partition needs a value}"; shift 2 ;;
      --time) time_limit="${2:?--time needs a value}"; shift 2 ;;
      --wait) wait_flag="--wait"; shift ;;
      --force) force="true"; shift ;;
      --dry-run) dry_run="true"; shift ;;
      -h|--help) sed -n '4,19p' "$0"; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
  done

  if [[ -z "${USER_NAME:-}" ]]; then
    echo "ERROR: USER_NAME is not set. Export your cluster user name first." >&2
    exit 1
  fi
  # shellcheck source=script/slurm/env_variables.sh
  source "${SCRIPT_DIR}/env_variables.sh"

  local images_dir name final_image uv_release
  images_dir="${NSS_SHARED_DIR}/images"
  name="$(image_name)"
  final_image="${images_dir}/${name}"
  uv_release="$(pinned_uv_release)" || {
    echo "ERROR: could not read the linux-x64 uv release from mise.lock" >&2
    exit 1
  }

  if [[ -e "${final_image}" && -z "${force}" ]]; then
    echo "Image already exists: ${final_image}"
    exit 0
  fi
  if [[ ! -f "${NSS_BASE_IMAGE}" ]]; then
    echo "ERROR: base image not found: ${NSS_BASE_IMAGE}" >&2
    exit 1
  fi

  local log_dir="${BASE_LOG_DIR}/image_builds"
  local sbatch_args=(
    --parsable
    --job-name nss_build_image
    --account "${ACCOUNT:-nemotron_data_dev}"
    --partition "${partition}"
    --time "${time_limit}"
    --nodes 1
    --ntasks 1
    --cpus-per-task 16
    --mem 64G
    --output "${log_dir}/build_%j.out"
    --error "${log_dir}/build_%j.err"
    --export "ALL,REPO_ROOT=${REPO_ROOT},NSS_IMAGES_DIR=${images_dir},NSS_IMAGE_NAME=${name},NSS_UV_URL=${uv_release% *},NSS_UV_SHA256=${uv_release#* }"
  )
  [[ -n "${wait_flag}" ]] && sbatch_args+=("${wait_flag}")

  echo "Building ${final_image}"
  echo "  from checkout ${REPO_ROOT} ($(git -C "${REPO_ROOT}" rev-parse --short HEAD))"
  echo "  base image   ${NSS_BASE_IMAGE}"
  if [[ -n "${dry_run}" ]]; then
    echo "--dry-run: sbatch ${sbatch_args[*]} ${SCRIPT_DIR}/build_image.sh --job"
    exit 0
  fi

  mkdir -p "${log_dir}"
  local job_id
  job_id="$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/build_image.sh" --job)"
  echo "Submitted build job ${job_id}; log: ${log_dir}/build_${job_id}.out"
  if [[ -n "${wait_flag}" ]]; then
    [[ -f "${final_image}" ]] || { echo "ERROR: build finished without ${final_image}" >&2; exit 1; }
    echo "Image ready: ${final_image}"
  fi
  echo "To use it, set NSS_IMAGE in env_variables.sh to that path."
}

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

  if [[ -e "${final_image}" ]]; then
    echo "Another build already published ${final_image}; discarding this one."
    exit 0
  fi
  chmod 664 "${tmp_image}"
  mv "${tmp_image}" "${final_image}"
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
  *) submit_build "$@" ;;
esac

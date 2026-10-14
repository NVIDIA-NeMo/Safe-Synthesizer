#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download pretrained models once into the read-only shared models directory
# (NSS_MODELS_DIR, default $NSS_SHARED_DIR/models). Jobs copy a model from
# there to node-local disk instead of each downloading it into a cache.
#
# Usage (login node):
#   bash script/slurm/download_models.sh [--partition P] [--wait] [MODEL_ID ...]
#
# With no MODEL_ID, downloads every training.pretrained_model named in
# script/slurm/configs/*.yaml. Existing models are skipped. Models keep the
# Hugging Face repo layout (<org>/<name>) because NSS picks model metadata by
# matching names such as "Mistral" in the path. Files matching
# NSS_MODEL_EXCLUDES (default: duplicate consolidated or .pth/.bin weights)
# are skipped.
#
# Modes: submit (default), --job (batch step), --container (inside the NSS image).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
readonly SCRIPT_DIR REPO_ROOT

configured_models() {
  sed -n 's/^[[:space:]]*pretrained_model:[[:space:]]*["'\'']\{0,1\}\([^"'\'' #]*\).*/\1/p' \
    "${REPO_ROOT}/script/slurm/configs/"*.yaml | sort -u
}

submit_download() {
  local partition="cpu_short" wait_flag=""
  local models=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --partition) partition="${2:?--partition needs a value}"; shift 2 ;;
      --wait) wait_flag="--wait"; shift ;;
      -h|--help) sed -n '4,20p' "$0"; exit 0 ;;
      -*) echo "Unknown argument: $1" >&2; exit 1 ;;
      *) models+=("$1"); shift ;;
    esac
  done
  if [[ ${#models[@]} -eq 0 ]]; then
    mapfile -t models < <(configured_models)
  fi

  if [[ -z "${USER_NAME:-}" ]]; then
    echo "ERROR: USER_NAME is not set. Export your cluster user name first." >&2
    exit 1
  fi
  # shellcheck source=script/slurm/env_variables.sh
  source "${SCRIPT_DIR}/env_variables.sh"

  if ! mkdir -p "${NSS_MODELS_DIR}" 2>/dev/null || [[ ! -w "${NSS_MODELS_DIR}" ]]; then
    echo "ERROR: cannot write to NSS_MODELS_DIR=${NSS_MODELS_DIR}" >&2
    echo "Export NSS_MODELS_DIR to a directory you own, or ask its owner to create it." >&2
    exit 1
  fi
  if [[ ! -f "${NSS_IMAGE}" ]]; then
    echo "ERROR: NSS image not found: ${NSS_IMAGE}; build it with build_image.sh" >&2
    exit 1
  fi

  local log_dir="${BASE_LOG_DIR}/model_downloads"
  mkdir -p "${log_dir}"
  local sbatch_args=(
    --parsable
    --job-name nss_download_models
    --account "${ACCOUNT:-nemotron_data_dev}"
    --partition "${partition}"
    --time 01:00:00
    --nodes 1
    --ntasks 1
    --cpus-per-task 8
    --mem 32G
    --output "${log_dir}/download_%j.out"
    --error "${log_dir}/download_%j.err"
    --export "ALL,REPO_ROOT=${REPO_ROOT},NSS_DOWNLOAD_MODELS=${models[*]}"
  )
  [[ -n "${wait_flag}" ]] && sbatch_args+=("${wait_flag}")

  echo "Downloading to ${NSS_MODELS_DIR}: ${models[*]}"
  local job_id
  job_id="$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/download_models.sh" --job)"
  echo "Submitted download job ${job_id}; log: ${log_dir}/download_${job_id}.out"
}

run_download_job() {
  : "${NSS_IMAGE:?}" "${NSS_MODELS_DIR:?}"
  srun --ntasks=1 \
    --container-image="${NSS_IMAGE}" \
    --no-container-mount-home \
    --container-mounts="/lustre:/lustre,${REPO_ROOT}:${REPO_ROOT}:ro" \
    bash "${REPO_ROOT}/script/slurm/download_models.sh" --container
}

download_in_container() {
  : "${NSS_DOWNLOAD_MODELS:?}" "${NSS_MODELS_DIR:?}" "${LUSTRE_DIR:?}"
  # Group-writable, so other team members can update or remove shared models.
  umask 002
  # HF_TOKEN for gated models such as Mistral.
  # shellcheck source=/dev/null
  [[ -f "${LUSTRE_DIR}/.api_tokens.sh" ]] && source "${LUSTRE_DIR}/.api_tokens.sh"
  # The Hugging Face cache stays in the container's local filesystem.
  export HF_HOME=/tmp/hf PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
  # read -a splits on whitespace without expanding the glob patterns.
  local excludes=()
  read -r -a excludes <<< "${NSS_MODEL_EXCLUDES:-consolidated*.safetensors original/* *.pth *.bin}"

  local model_id dest tmp
  for model_id in ${NSS_DOWNLOAD_MODELS}; do
    dest="${NSS_MODELS_DIR}/${model_id}"
    if [[ -d "${dest}" ]]; then
      echo "skip ${model_id}: already at ${dest}"
      continue
    fi
    tmp="${dest}.tmp-${SLURM_JOB_ID:-$$}"
    mkdir -p "$(dirname "${dest}")"
    echo "download ${model_id} -> ${dest}"
    # hf takes one pattern per --exclude flag; extra bare arguments would be read as filenames.
    local exclude_args=() pattern
    for pattern in "${excludes[@]}"; do
      exclude_args+=(--exclude "${pattern}")
    done
    hf download "${model_id}" --local-dir "${tmp}" "${exclude_args[@]}"
    if [[ ! -d "${tmp}" ]] || ! compgen -G "${tmp}/*.safetensors" > /dev/null; then
      echo "ERROR: ${model_id} downloaded no .safetensors weights into ${tmp}" >&2
      rm -rf "${tmp}"
      exit 1
    fi
    # hf leaves download metadata and lock files under .cache; jobs only need the model files.
    rm -rf "${tmp}/.cache"
    mv -T "${tmp}" "${dest}"
    du -sh "${dest}"
  done
}

case "${1:-}" in
  --job) run_download_job ;;
  --container) download_in_container ;;
  *) submit_download "$@" ;;
esac

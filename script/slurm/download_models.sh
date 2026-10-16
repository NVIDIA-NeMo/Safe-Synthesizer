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
# Submitting is handled by nss_slurm.py ("download-models"). This script keeps
# the compute-node steps: --job (batch step) and --container (inside the NSS image).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
readonly SCRIPT_DIR REPO_ROOT

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
  *) exec bash "${SCRIPT_DIR}/nss_slurm.sh" download-models "$@" ;;
esac

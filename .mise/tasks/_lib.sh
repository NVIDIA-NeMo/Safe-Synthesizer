# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash

resolve_python_version() {
  local python_version

  if [[ -n "${PYTHON_VERSION:-}" ]]; then
    python_version="${PYTHON_VERSION}"
  elif [[ -f "${MISE_CONFIG_ROOT}/.python-version" ]]; then
    python_version="$(tr -d '[:space:]' < "${MISE_CONFIG_ROOT}/.python-version")"
  else
    echo "Error: PYTHON_VERSION is unset and ${MISE_CONFIG_ROOT}/.python-version does not exist" >&2
    return 1
  fi

  if [[ -z "${python_version}" ]]; then
    echo "Error: Python version is empty" >&2
    return 1
  fi

  printf '%s\n' "${python_version}"
}

resolve_venv_path() {
  local venv_path="${UV_PROJECT_ENVIRONMENT:-${MISE_CONFIG_ROOT:?MISE_CONFIG_ROOT is required}/.venv}"

  if [[ -z "${venv_path}" \
    || "${venv_path}" == "/" \
    || "${venv_path}" == "." \
    || "${venv_path}" == ".." \
    || "${venv_path}" == "${MISE_CONFIG_ROOT}" \
    || "${venv_path}" == "${MISE_CONFIG_ROOT}/" ]]; then
    echo "Error: refusing to use invalid virtualenv path: '${venv_path}'" >&2
    return 1
  fi

  printf '%s\n' "${venv_path}"
}

sync_nss_dependencies() {
  local extra="${1:?Python dependency profile is required}"
  local python="${2:-}"
  local sync_uv_bin="${NSS_UV_BIN:-uv}"

  if [[ -z "${python}" ]]; then
    python="$(resolve_python_version)" || return
  fi

  case "${extra}" in
    cuda|cu129)
      "${sync_uv_bin}" sync --frozen --python "${python}" --extra cu129 --extra engine --group dev
      ;;
    cpu)
      "${sync_uv_bin}" sync --frozen --python "${python}" --extra cpu --extra engine --group dev
      ;;
    engine)
      "${sync_uv_bin}" sync --frozen --python "${python}" --extra engine --group dev
      ;;
    dev)
      "${sync_uv_bin}" sync --frozen --python "${python}" --group dev
      ;;
    *)
      echo "Error: Invalid extra '${extra}'. Use one of: dev engine cpu cuda cu129" >&2
      return 1
      ;;
  esac
}

resolve_container_cmd() {
  local container_cmd="${CONTAINER_CMD:-$(command -v podman 2>/dev/null || command -v docker 2>/dev/null || true)}"

  if [[ -z "${container_cmd}" ]]; then
    echo "Error: neither podman nor docker was found" >&2
    return 1
  fi

  printf '%s\n' "${container_cmd}"
}

split_words() {
  local -n output_array="$1"
  local value="${2:-}"

  output_array=()
  if [[ -n "${value}" ]]; then
    read -r -a output_array <<< "${value}"
  fi
}

ensure_hf_cache() {
  local hf_cache="${CONTAINER_HF_CACHE:-${HOME}/.cache/huggingface}"

  if ! mkdir -p "${hf_cache}"; then
    echo "Error: failed to create Hugging Face cache directory: '${hf_cache}'" >&2
    return 1
  fi
  printf '%s\n' "${hf_cache}"
}

require_env() {
  local name="$1"
  local message="$2"

  if [[ -z "${!name:-}" ]]; then
    echo "Error: ${message}" >&2
    return 1
  fi
}

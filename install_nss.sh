#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly PACKAGE_NAME="${PACKAGE_NAME:-nemo-safe-synthesizer}"
readonly PACKAGE_VERSION="${PACKAGE_VERSION:-}"
readonly CUDA="${CUDA:-129}"
readonly INSTALLER="${INSTALLER:-uv}"
readonly DRY_RUN="${DRY_RUN:-0}"
readonly CONSTRAINTS_URL="${CONSTRAINTS_URL:-https://raw.githubusercontent.com/NVIDIA-NeMo/Safe-Synthesizer/main/constraints.txt}"
readonly NVIDIA_DRIVER_MIN_CUDA_13="580.65.06"

usage() {
    cat <<'EOF'
Install NeMo Safe Synthesizer with the right runtime indexes.

Usage:
  CUDA=129 bash install_nss.sh
  CUDA=130 bash install_nss.sh
  CUDA=cpu bash install_nss.sh

Environment:
  CUDA=129|130|cpu|help   Runtime extra to install. Default: 129.
  INSTALLER=uv|pip        Installer command. Default: uv.
  PACKAGE_VERSION=<spec>  Optional version specifier, for example ==0.1.0.
  CONSTRAINTS_URL=<url>   Constraints file URL. Default: repo main constraints.txt.
  DRY_RUN=1               Print the command without running it.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

warn() {
    echo "WARNING: $*" >&2
}

require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || die "'${command_name}' is required but was not found on PATH"
}

version_at_least() {
    local current="$1"
    local minimum="$2"
    local first
    first="$(printf '%s\n%s\n' "$minimum" "$current" | sort -V | head -n 1)"
    [[ "$first" == "$minimum" ]]
}

driver_version() {
    { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true; } | head -n 1 | tr -d '[:space:]'
}

check_cuda_13_driver() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi not found; CUDA 13.x requires Linux NVIDIA driver ${NVIDIA_DRIVER_MIN_CUDA_13}+"
        return
    fi

    local current
    current="$(driver_version)"
    if [[ -z "$current" ]]; then
        warn "could not read NVIDIA driver version; CUDA 13.x requires ${NVIDIA_DRIVER_MIN_CUDA_13}+"
        return
    fi

    if ! version_at_least "$current" "$NVIDIA_DRIVER_MIN_CUDA_13"; then
        die "CUDA 13.x requires Linux NVIDIA driver ${NVIDIA_DRIVER_MIN_CUDA_13}+; found ${current}"
    fi
}

runtime_extra() {
    case "$CUDA" in
        129 | cu129 | cuda12 | cuda12.9) printf 'cu129' ;;
        130 | cu130 | cuda13 | cuda13.0) printf 'cu130' ;;
        cpu | CPU) printf 'cpu' ;;
        -h | --help | help)
            usage
            exit 0
            ;;
        *) die "unsupported CUDA='${CUDA}'. Use 129, 130, or cpu." ;;
    esac
}

runtime_indexes() {
    local extra="$1"
    case "$extra" in
        cu129)
            INDEXES=(
                "https://flashinfer.ai/whl/cu129"
                "https://download.pytorch.org/whl/cu129"
                "https://pypi.nvidia.com"
            )
            ;;
        cu130)
            INDEXES=(
                "https://flashinfer.ai/whl/cu130"
                "https://download.pytorch.org/whl/cu130"
                "https://pypi.nvidia.com"
            )
            ;;
        cpu)
            INDEXES=()
            if [[ "$(uname -s)" == "Linux" ]]; then
                INDEXES=("https://download.pytorch.org/whl/cpu")
            fi
            ;;
    esac
}

package_spec() {
    local extra="$1"
    printf '%s[%s]%s' "$PACKAGE_NAME" "$extra" "$PACKAGE_VERSION"
}

append_common_args() {
    INSTALL_CMD+=("-c" "$CONSTRAINTS_URL")
}

append_index_args() {
    local flag="$1"
    local index
    for index in "${INDEXES[@]}"; do
        INSTALL_CMD+=("$flag" "$index")
    done
}

build_install_command() {
    local extra="$1"
    runtime_indexes "$extra"
    case "$INSTALLER" in
        uv)
            require_command uv
            INSTALL_CMD=(uv pip install "$(package_spec "$extra")")
            append_common_args
            append_index_args "--index"
            if [[ "$extra" != "cpu" ]]; then
                INSTALL_CMD+=("--index-strategy" "unsafe-best-match")
            fi
            ;;
        pip)
            require_command pip
            if [[ "$extra" != "cpu" ]]; then
                warn "pip prefers PyPI over extra indexes; use INSTALLER=uv for CUDA installs when possible."
            fi
            INSTALL_CMD=(pip install "$(package_spec "$extra")")
            append_common_args
            append_index_args "--extra-index-url"
            ;;
        *) die "unsupported INSTALLER='${INSTALLER}'. Use uv or pip." ;;
    esac
}

run_install() {
    printf 'Installing with:'
    printf ' %q' "${INSTALL_CMD[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi
    "${INSTALL_CMD[@]}"
}

main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
        usage
        exit 0
    fi
    if [[ "$CUDA" == "-h" || "$CUDA" == "--help" || "$CUDA" == "help" ]]; then
        usage
        exit 0
    fi

    local extra
    extra="$(runtime_extra)"
    if [[ "$extra" == "cu130" && "$DRY_RUN" != "1" ]]; then
        check_cuda_13_driver
    fi
    build_install_command "$extra"
    run_install
}

INDEXES=()
INSTALL_CMD=()
main "$@"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly PACKAGE_NAME="${PACKAGE_NAME:-nemo-safe-synthesizer}"
readonly PACKAGE_VERSION="${PACKAGE_VERSION:-}"
readonly CUDA="${CUDA:-129}"
readonly DRY_RUN="${DRY_RUN:-0}"
readonly CONSTRAINTS_URL="${CONSTRAINTS_URL:-https://raw.githubusercontent.com/NVIDIA-NeMo/Safe-Synthesizer/main/constraints.txt}"
readonly NVIDIA_DRIVER_MIN_CUDA_13="580.65.06"

usage() {
    cat <<'EOF'
Install NeMo Safe Synthesizer with a supported runtime extra.

Usage:
  CUDA=129 ./install_nss.sh
  CUDA=130 ./install_nss.sh
  CUDA=cpu ./install_nss.sh

Environment:
  CUDA=129|130|cpu|help   Runtime extra to install. Default: 129.
  PACKAGE_VERSION=<spec>  Optional version specifier, for example ==0.1.0.
  CONSTRAINTS_URL=<url>   Constraints file URL. Default: the main branch.
  DRY_RUN=1               Print the uv command without running it.
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
    command -v "$1" >/dev/null 2>&1 || die "'$1' is required but was not found on PATH"
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
        die "CUDA 13.x requires Linux NVIDIA driver ${NVIDIA_DRIVER_MIN_CUDA_13}+; found $current"
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
    case "$1" in
        cu129)
            INDEXES=(
                "https://flashinfer.ai/whl/cu129"
                "https://download.pytorch.org/whl/cu129"
                "https://wheels.vllm.ai/ee0da84ab9e04ac7610e28580af62c365e898389/cu129"
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
    printf '%s[engine,%s]%s' "$PACKAGE_NAME" "$1" "$PACKAGE_VERSION"
}

build_install_command() {
    local extra="$1"
    local index

    runtime_indexes "$extra"
    INSTALL_CMD=(uv pip install "$(package_spec "$extra")" -c "$CONSTRAINTS_URL")
    for index in "${INDEXES[@]}"; do
        INSTALL_CMD+=(--index "$index")
    done
    if [[ "$extra" != "cpu" ]]; then
        INSTALL_CMD+=(--index-strategy unsafe-best-match)
    fi
}

run_install() {
    printf 'Installing with:'
    printf ' %q' "${INSTALL_CMD[@]}"
    printf '\n'

    [[ "$DRY_RUN" == "1" ]] || "${INSTALL_CMD[@]}"
}

main() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
        usage
        return
    fi

    local extra
    extra="$(runtime_extra)"
    require_command uv
    if [[ "$extra" == "cu130" && "$DRY_RUN" != "1" ]]; then
        check_cuda_13_driver
    fi
    build_install_command "$extra"
    run_install
}

INDEXES=()
INSTALL_CMD=()
main "$@"

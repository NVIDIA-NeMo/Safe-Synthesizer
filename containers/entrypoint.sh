#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper entrypoint for the nss-gpu runtime container.
# Detects common misconfigurations and prints hints before
# passing through to safe-synthesizer.

set -euo pipefail

warn() { echo "[nss-container] $*" >&2; }

# Skip all diagnostic checks for info-only commands.
needs_runtime=true
for arg in "$@"; do
    case "$arg" in
        --help|-h|--version|-V) needs_runtime=false; break ;;
        config)                  needs_runtime=false; break ;;
    esac
done

if $needs_runtime; then

# -- Check: is anything mounted at /workspace?
if [ -z "$(ls -A /workspace 2>/dev/null)" ]; then
    warn "WARNING: /workspace is empty -- no data or config files are available."
    echo >&2
    warn "  Mount your data directory (use absolute paths):"
    warn "    docker run --gpus all -v /path/to/data:/workspace/data \\"
    warn "      -v ~/.cache/huggingface:/workspace/.hf_cache \\"
    warn "      -e HF_HOME=/workspace/.hf_cache \\"
    warn "      nss-gpu:latest run --config /workspace/data/config.yaml --url /workspace/data/input.csv"
    echo >&2
    warn "  Relative paths don't work with docker -v. Use \$(pwd) to expand them:"
    warn "    -v \$(pwd)/my_data:/workspace/data"
    echo >&2
fi

# -- Check: is HF_HOME set and does it exist?
if [ -n "${HF_HOME:-}" ] && [ ! -d "$HF_HOME" ]; then
    warn "WARNING: HF_HOME=$HF_HOME does not exist. Model downloads will use a temporary directory"
    warn "  and be lost when the container exits."
    echo >&2
    warn "  Mount your Hugging Face cache:"
    warn "    -v ~/.cache/huggingface:/workspace/.hf_cache -e HF_HOME=/workspace/.hf_cache"
    echo >&2
elif [ -z "${HF_HOME:-}" ]; then
    warn "HINT: HF_HOME is not set. Models will download to a temporary location."
    warn "  To persist model downloads across runs:"
    warn "    -v ~/.cache/huggingface:/workspace/.hf_cache -e HF_HOME=/workspace/.hf_cache"
    echo >&2
fi

# -- Check: HF_TOKEN for gated model access (Llama, Mistral, etc.)
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
    if [ ! -f "$token_file" ]; then
        warn "HINT: No HF_TOKEN set and no cached token found."
        warn "  Gated models (Llama, Mistral) will fail to download."
        warn "  Pass your token:  -e HF_TOKEN=\"hf_...\""
        warn "  Or mount a cache that contains one:  -v ~/.cache/huggingface:/workspace/.hf_cache"
        echo >&2
    fi
fi

# -- Check: can we see NVIDIA GPUs?
if ! command -v nvidia-smi &>/dev/null; then
    warn "WARNING: nvidia-smi not found. GPU access may not be available."
    warn "  Run with: docker run --gpus all ..."
    echo >&2
fi

# -- Check: /dev/shm too small for PyTorch DataLoader multi-worker IPC.
shm_kb=$(df -k /dev/shm 2>/dev/null | awk 'NR==2 {print $2}') || shm_kb=""
if [ -n "$shm_kb" ] && [ "$shm_kb" -lt 262144 ]; then
    warn "WARNING: /dev/shm is only $((shm_kb / 1024)) MB. Training with multi-worker"
    warn "  data loading may crash with 'Bus error'. Increase shared memory:"
    warn "    docker run --shm-size=1g ...   # or --ipc=host"
    echo >&2
fi

fi  # needs_runtime

exec safe-synthesizer "$@"

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


export USER_NAME="${USER_NAME:-}"
export LUSTRE_DIR="/lustre/fsw/portfolios/llmservice/users/${USER_NAME}" ## do not change this

# NOTE: This directory is only setup on cs-oci-ord and cw-dfw-cs right now,
# will need to create the same structure on other clusters as we need them.
export NSS_SHARED_DIR="/lustre/fsw/portfolios/llmservice/users/kendrickb/shared_safe_synthesizer"

## change the followings if you want them to be different
CONFIGS=(unsloth dp dp_usg_guidance) # the jobs will run all datasets with these configs
export NSS_DIR="/lustre/fsw/portfolios/llmservice/users/${USER_NAME}/Safe-Synthesizer" # where the nss repo is located
export NSS_SLURM_DIR="${NSS_DIR}/script/slurm" # slurm scripts location (inside repo)
export CONFIG_DIR="${NSS_SLURM_DIR}" # where the config files are located
export BASE_LOG_DIR="${LUSTRE_DIR}/nss_results" # where you want the slurm logs to be saved, each job will have err and out files
export ADAPTER_PATH="${LUSTRE_DIR}/nmp/exp/adapters" # base path for run directories (each run creates a subdirectory via --run-path)
export VLLM_CACHE_ROOT="${LUSTRE_DIR}/.cache/vllm/" # where the vllm cache is saved, this is to prevent the login node from blowing up
export UV_CACHE_DIR="${LUSTRE_DIR}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${LUSTRE_DIR}/.local/share/uv/python"
export UV_PYTHON_BIN_DIR="${LUSTRE_DIR}/.local/bin"
export UV_TOOL_DIR="${LUSTRE_DIR}/.local/share/uv/tools"
export HF_HOME="${LUSTRE_DIR}/.cache/huggingface"
export WANDB_MODE="disabled" # "online", "offline" or "disabled"

# NSS CLI environment variables (used by safe-synthesizer CLI via pydantic-settings)
# These are picked up automatically by CLISettings in the CLI:
#   NSS_ARTIFACTS_PATH - Base directory for artifacts (same as ADAPTER_PATH)
#   NSS_PHASE - Current phase (train, generate, end_to_end)
#   NSS_CONFIG - Path to YAML config file
#   NSS_LOG_FORMAT - Log format ("json" or "plain")
#   NSS_LOG_FILE - Path to log file
export NSS_ARTIFACTS_PATH="${ADAPTER_PATH}"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Storage follows the MARS cluster rules: source code in /home, images, data,
# and outputs on Lustre, and every cache on node-local /raid/scratch. Caches
# are set per job in slurm_nss_matrix.sh, not here.

export USER_NAME="${USER_NAME:-}"
export LUSTRE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/${USER_NAME}" ## do not change this

# Team-shared images, data, and models, owned by group llmservice (setgid, 2775).
# NOTE: This directory is only set up on cs-oci-ord right now; create the same
# structure on other clusters as we need them (see README).
export NSS_SHARED_DIR="${NSS_SHARED_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer}"
# Base image for build_image.sh; jobs run from the NSS image built on top of it.
export NSS_BASE_IMAGE="${NSS_BASE_IMAGE:-${NSS_SHARED_DIR}/images/cuda_12_8_1_cudnn_runtime_ubuntu24_04.sqsh}"
# Team-shared NSS image from build_image.sh. Bump this when a new image is built.
export NSS_IMAGE="${NSS_IMAGE:-${NSS_SHARED_DIR}/images/nss-slurm-py3.13-e58529b3b46c.sqsh}"
# Read-only, pre-downloaded models (<org>/<name>), staged to node-local disk by jobs.
export NSS_MODELS_DIR="${NSS_MODELS_DIR:-${NSS_SHARED_DIR}/models}"

## change the followings if you want them to be different
# shellcheck disable=SC2034 # read by nss_slurm.sh after sourcing this file
CONFIGS=(
    smollm3-nodp
    smollm3-dp
    tinyllama-nodp
    tinyllama-dp
    mistral-nodp
    mistral-dp
) # the jobs will run all datasets with these configs
# Repository checkout in /home that contains this script; jobs mount it at the same path.
_nss_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NSS_DIR="${NSS_DIR:-$(cd "${_nss_env_dir}/../.." && pwd)}"
unset _nss_env_dir
export NSS_SLURM_DIR="${NSS_DIR}/script/slurm" # slurm scripts location (inside repo)
export CONFIG_DIR="${NSS_SLURM_DIR}/configs" # where the config files are located
export BASE_LOG_DIR="${LUSTRE_DIR}/nss_results" # where you want the slurm logs to be saved, each job will have err and out files
export ADAPTER_PATH="${LUSTRE_DIR}/nss_results/adapters" # base path for run directories (each run creates a subdirectory via --run-path)
export WANDB_MODE="${WANDB_MODE:-online}" # "online", "offline" or "disabled"; an exported value wins
export NEMO_DEPLOYMENT_TYPE="slurm-nvidia-internal" # sets the values for telemetry

# NSS CLI environment variables (used by safe-synthesizer CLI via pydantic-settings)
# These are picked up automatically by CLISettings in the CLI:
#   NSS_ARTIFACTS_PATH - Base directory for artifacts (same as ADAPTER_PATH)
#   NSS_PHASE - Current phase (train, generate, end_to_end)
#   NSS_CONFIG - Path to YAML config file
#   NSS_LOG_FORMAT - Log format ("json" or "plain")
#   NSS_LOG_FILE - Path to log file
export NSS_ARTIFACTS_PATH="${ADAPTER_PATH}"

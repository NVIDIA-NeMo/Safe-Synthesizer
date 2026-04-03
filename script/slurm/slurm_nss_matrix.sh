#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

# The following environment variables are expected to be set. Some have
# plausible default values, and others we check and throw an immediate
# error here at the top of the script. This list of environment variables
# and checks also serves as documentation for the script.

# This script is intended to be executed through slurm, so these environment
# variables would be set automatically as part of the slurm job submission.
# But being able to run this script standalone for debugging purposes is
# useful.

# If running standalone, manually source env_variables.sh to pick up many
# needed environment variables.

if [ -z "${PACKED_DATASETS:-}" ]; then
    echo "PACKED_DATASETS must be set" >&2
    echo "Run script through submit_slurm_jobs.sh, or if running manually, export PACKED_DATASETS with a single dataset name or url/path" >&2
    exit 1
fi

if [ -z "${PACKED_CONFIGS:-}" ]; then
    echo "PACKED_CONFIGS must be set" >&2
    echo "Run script through submit_slurm_jobs.sh, or if running manually, export PACKED_CONFIGS with a single config name (without .yaml extension) to be found in the CONFIG_DIR" >&2
    exit 1
fi

if [ -z "${NSS_DIR:-}" ]; then
    echo "NSS_DIR must be set" >&2
    echo "Run script through submit_slurm_jobs.sh, or if running manually source env_variables.sh before running" >&2
    exit 1
fi

if [ -z "${LUSTRE_DIR:-}" ]; then
    echo "LUSTRE_DIR must be set" >&2
    echo "Run script through submit_slurm_jobs.sh, or if running manually source env_variables.sh before running" >&2
    exit 1
fi

if [[ ! -f "${LUSTRE_DIR}/.api_tokens.sh" ]]; then
    echo "ERROR: ${LUSTRE_DIR}/.api_tokens.sh not found." >&2
    echo "Create it with at least NSS_INFERENCE_KEY (and WANDB_API_KEY if WANDB_MODE=online)." >&2
    exit 1
fi
source "${LUSTRE_DIR}/.api_tokens.sh"

if [[ "${WANDB_MODE:-disabled}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_MODE is 'online' but WANDB_API_KEY is not set." >&2
    echo "Add 'export WANDB_API_KEY=\"<your_key>\"' to ${LUSTRE_DIR}/.api_tokens.sh" >&2
    echo "Or set WANDB_MODE=disabled in env_variables.sh to skip W&B logging." >&2
    exit 1
fi

if [ -z "${NSS_SHARED_DIR:-}" ]; then
    echo "NSS_SHARED_DIR must be set" >&2
    echo "Run script through submit_slurm_jobs.sh, or if running manually source env_variables.sh before running" >&2
    exit 1
fi
SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
SLURM_JOB_ID="${SLURM_JOB_ID:-0}"
NSS_PHASE="${NSS_PHASE:-end_to_end}"
# EXP_NAME controls the experiment namespace under nss_results
EXP_NAME="${EXP_NAME:-multi_jobs}"
BASE_LOG_DIR="${BASE_LOG_DIR:-./}"


cd "${NSS_DIR}"

# Get dataset and config name from packed strings
declare -a all_datasets
declare -a all_configs
IFS=',' read -r -a all_datasets <<< "${PACKED_DATASETS}"
IFS=',' read -r -a all_configs <<< "${PACKED_CONFIGS}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    dataset=${all_datasets[$SLURM_ARRAY_TASK_ID]}
    config=${all_configs[$SLURM_ARRAY_TASK_ID]}
else
    echo "SLURM_ARRAY_TASK_ID is not set, assuming single dataset and config"
    if [ ${#all_datasets[@]} -ne 1 ] || [ ${#all_configs[@]} -ne 1 ]; then
        echo "Exactly one dataset and one config must be provided when SLURM_ARRAY_TASK_ID is not set" >&2
        exit 1
    fi
    dataset=${all_datasets[0]}
    config=${all_configs[0]}
fi

# Ensure minimal build toolchain inside container (no-op if already present)
apt-get update && apt-get install -y --no-install-recommends \
        curl \
        g++ \
        build-essential \
        ca-certificates \
        git \
        libcurl4 \
        libcurl3-gnutls

# Ensure Python environment is available inside the container
source "${LUSTRE_DIR}/.uv/bin/env"
source "${NSS_DIR}/.venv/bin/activate"
uv sync --frozen --extra cu128 --extra engine --group dev


# for column classification
export NSS_INFERENCE_ENDPOINT=https://integrate.api.nvidia.com/v1
export NIM_MODEL_ID=qwen/qwen2.5-coder-32b-instruct

# Extract dataset name for path construction (handles both full paths and simple names)
# e.g., "/path/to/adult.csv" -> "adult", "/path/to/data.parquet" -> "data", "adult" -> "adult"
# Supports: .csv, .parquet, .json, .jsonl
dataset_basename=$(basename "$dataset")
dataset_name="${dataset_basename%.*}"

full_config_path="${CONFIG_DIR}/${config}.yaml"
if [[ ! -f "$full_config_path" ]]; then
    echo "Config file not found: ${full_config_path}" >&2
    exit 1
fi

# Construct run path for WorkdirStructure
# - two_stage mode (train/generate): Share same base path without SLURM ID
#   Note: Don't submit multiple batches of same config/dataset/job_idx concurrently in two_stage mode
#   The train phase creates the workdir, generate phase resumes from it (with unique output files)
# - end_to_end mode: Include SLURM ID for uniqueness across concurrent runs
slurm_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    slurm_id="${slurm_id}_${SLURM_ARRAY_TASK_ID}"
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    job_idx=${SLURM_ARRAY_TASK_ID}
else
    job_idx=0
fi


if [[ "${NSS_PHASE}" == "end_to_end" ]]; then
    # end_to_end: include SLURM ID for unique paths across concurrent runs
    run_path=${ADAPTER_PATH}/${config}_${dataset_name}_${job_idx}_${slurm_id}
else
    # two_stage (train/generate): shared base path without SLURM ID
    # Train creates workdir; generate resumes and creates unique output files per run
    run_path=${ADAPTER_PATH}/${config}_${dataset_name}_${job_idx}
fi

# Results directory (optional; experiments may also write their own outputs)

OUTPUT_DIR="${BASE_LOG_DIR}/${EXP_NAME}/${dataset_name}/${config}/run_${job_idx}"
# mkdir -p "$OUTPUT_DIR"

# Derive a stable seed unless overridden
SEED=${SEED_OVERRIDE:-$(( 42 +  job_idx ))}
export SEED


echo "[NSS SLURM] config=${config}, dataset=${dataset_name}, job_idx=${job_idx}, slurm_id=${slurm_id}"
echo "[NSS SLURM] full_config_path=${full_config_path}"
echo "[NSS SLURM] full dataset path/name=${dataset}"
echo "[NSS SLURM] run_path=$run_path"
echo "[NSS SLURM] output_dir=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES is set to: ${CUDA_VISIBLE_DEVICES:-not set}"

dataset_registry_arg=""
dataset_registry_file="${NSS_SHARED_DIR}/dataset_registry.yaml"
if [ -f "${dataset_registry_file}" ]; then
    dataset_registry_arg="--dataset-registry ${dataset_registry_file}"
    echo "Using dataset registry file: ${dataset_registry_file}"
else
    echo "No dataset registry file found at ${dataset_registry_file}"
fi


if [[ "${NSS_PHASE}" == "train" ]]; then
    # Stage 1: PII replacement + training
    # Creates new workdir at run_path with adapter
    uv run safe-synthesizer run train \
        --data-source "$dataset" \
        --config "$full_config_path" \
        --run-path "$run_path" \
        $dataset_registry_arg
elif [[ "${NSS_PHASE}" == "generate" ]]; then
    # Stage 2: generation (+ optional evaluation)
    # Resumes from existing workdir at run_path
    # Each generation run creates unique output files (synthetic_data_TIMESTAMP.csv)
    # Resume wandb run from train phase if the ID file exists
    wandb_id_file="${run_path}/wandb_run_id.txt"
    wandb_resume_arg=""
    if [[ -f "$wandb_id_file" ]]; then
        wandb_resume_arg="--wandb-resume-job-id $wandb_id_file"
    fi

    uv run safe-synthesizer run generate \
        --data-source "$dataset" \
        --config "$full_config_path" \
        --run-path "$run_path" \
        $dataset_registry_arg \
        $wandb_resume_arg
else
    # Full end-to-end run
    uv run safe-synthesizer run \
        --data-source "$dataset" \
        --config "$full_config_path" \
        --run-path "$run_path" \
        $dataset_registry_arg
fi

echo "[NSS SLURM] Done"

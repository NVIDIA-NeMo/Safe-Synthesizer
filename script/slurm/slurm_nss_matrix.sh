#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

# The following environment variables are expected to be set. Some have
# plausible default values, and others we check and throw an immediate
# error here at the top of the script. This list of environment variables
# and checks also serves as documentation for the script.

# This script is intended to be executed through slurm, inside the NSS image
# built by build_image.sh, so these environment variables would be set
# automatically as part of the slurm job submission. But being able to run
# this script standalone for debugging purposes is useful.

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

for required in NSS_DIR LUSTRE_DIR NSS_SHARED_DIR USER_NAME; do
    if [ -z "${!required:-}" ]; then
        echo "${required} must be set" >&2
        echo "Run script through submit_slurm_jobs.sh, or if running manually source env_variables.sh before running" >&2
        exit 1
    fi
done

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

if [[ ! -x /opt/venv/bin/python || ! -f /opt/nss-image/.python-version ]]; then
    echo "[NSS SLURM] ERROR: not running inside an NSS image (no /opt/venv)." >&2
    echo "[NSS SLURM] Build one with script/slurm/build_image.sh and set NSS_IMAGE." >&2
    exit 1
fi

SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
SLURM_JOB_ID="${SLURM_JOB_ID:-0}"
NSS_PHASE="${NSS_PHASE:-end_to_end}"
# EXP_NAME controls the experiment namespace under nss_results
EXP_NAME="${EXP_NAME:-multi_jobs}"
BASE_LOG_DIR="${BASE_LOG_DIR:-./}"

# Node-local storage. The container runs as root mapped to the submitting
# user, so paths use USER_NAME rather than USER. NODE_CACHE persists across
# jobs on the same node; JOB_LOCAL is removed when the job exits.
NODE_CACHE="/raid/scratch/${USER_NAME}/nss-cache"
JOB_LOCAL="/raid/scratch/${USER_NAME}/job-${SLURM_JOB_ID}-$$"
mkdir -p "${NODE_CACHE}/uv" "${NODE_CACHE}/hf" "${NODE_CACHE}/models" "${JOB_LOCAL}/tmp"
run_path=""  # set below; read by cleanup_job_local
cleanup_job_local() {
    # Offline W&B runs exist only under WANDB_DIR. Keep them as one archive in
    # the run directory so they can be uploaded later with `wandb sync`.
    if [[ "${WANDB_MODE:-}" == "offline" && -n "${run_path}" && -d "${WANDB_DIR:-}/wandb" ]]; then
        mkdir -p "${run_path}"
        tar -C "${WANDB_DIR}" -cf "${run_path}/wandb_offline_${SLURM_JOB_ID}.tar" wandb \
            || echo "[NSS SLURM] WARNING: could not archive offline W&B files to ${run_path}" >&2
    fi
    rm -rf "${JOB_LOCAL}"
}
trap cleanup_job_local EXIT
trap 'exit 143' TERM INT

# Every cache, temp, and config directory points at node-local disk, and any
# values inherited from the login shell are overridden or removed.
unset XDG_RUNTIME_DIR UV_PYTHON_BIN_DIR UV_TOOL_DIR UV_TOOL_BIN_DIR VIRTUAL_ENV PYTHONHOME
export PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
export UV_CACHE_DIR="${NODE_CACHE}/uv"
export UV_PROJECT_ENVIRONMENT=/opt/venv
export UV_PYTHON_INSTALL_DIR=/opt/python
export UV_MANAGED_PYTHON=1
export UV_LINK_MODE=copy
export HF_HOME="${NODE_CACHE}/hf"
export VLLM_CACHE_ROOT="${JOB_LOCAL}/vllm"
export TRITON_CACHE_DIR="${JOB_LOCAL}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_LOCAL}/torchinductor"
export FLASHINFER_WORKSPACE_BASE="${JOB_LOCAL}/flashinfer"
export XDG_CACHE_HOME="${JOB_LOCAL}/xdg-cache"
export XDG_CONFIG_HOME="${JOB_LOCAL}/xdg-config"
export WANDB_DIR="${JOB_LOCAL}/wandb"
export WANDB_CACHE_DIR="${JOB_LOCAL}/wandb/cache"
export WANDB_CONFIG_DIR="${JOB_LOCAL}/wandb/config"
export WANDB_DATA_DIR="${JOB_LOCAL}/wandb/data"
export TMPDIR="${JOB_LOCAL}/tmp"
export PYTHONPYCACHEPREFIX="${JOB_LOCAL}/pycache"
mkdir -p "${WANDB_DIR}"

# Install the project (editable, from the read-only checkout) and any
# dependency differences between this branch's uv.lock and the image.
branch_python="$(tr -d '[:space:]' < "${NSS_DIR}/.python-version")"
image_python="$(tr -d '[:space:]' < /opt/nss-image/.python-version)"
if [[ "${branch_python}" != "${image_python}" ]]; then
    echo "[NSS SLURM] ERROR: branch needs Python ${branch_python} but the image has ${image_python}." >&2
    echo "[NSS SLURM] Build a new image with script/slurm/build_image.sh." >&2
    exit 1
fi
if cmp -s "${NSS_DIR}/uv.lock" /opt/nss-image/uv.lock; then
    echo "[NSS SLURM] uv.lock matches the image; installing the project only"
else
    echo "[NSS SLURM] uv.lock differs from the image; syncing dependency differences"
fi
(cd "${NSS_DIR}" && uv sync --frozen --no-dev --extra engine --extra cu129)
NSS_RUN_CMD="/opt/venv/bin/safe-synthesizer"
# Run from node-local disk so any relative-path writes stay off Lustre and
# off the read-only checkout.
cd "${JOB_LOCAL}"

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

echo "[NSS SLURM] nemo-safe-synthesizer version: $(python -c 'from nemo_safe_synthesizer.package_info import __version__; print(__version__)')"


# for column classification
export NSS_INFERENCE_ENDPOINT=https://integrate.api.nvidia.com/v1
export NSS_INFERENCE_MODEL=nvidia/nemotron-3-ultra-550b-a55b

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

# Stage the pretrained model from the read-only shared copy to node-local
# disk, once per node. The path keeps the Hugging Face repo name because NSS
# picks model metadata by matching names such as "Mistral" in the path.
model_args=()
model_id="$(python -c 'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1])) or {}; print((cfg.get("training") or {}).get("pretrained_model") or "")' "$full_config_path")"
if [[ -n "${model_id}" && "${model_id}" != /* && -d "${NSS_MODELS_DIR:-}/${model_id}" ]]; then
    staged_model="${NODE_CACHE}/models/${model_id}"
    if [[ ! -d "${staged_model}" ]]; then
        echo "[NSS SLURM] staging ${model_id} from ${NSS_MODELS_DIR} to ${staged_model}"
        mkdir -p "$(dirname "${staged_model}")"
        # Copy inside JOB_LOCAL so the exit trap removes a partial copy if the job
        # is cancelled or times out. JOB_LOCAL and NODE_CACHE are on the same
        # /raid/scratch filesystem, so the rename below is atomic.
        staging_tmp="${JOB_LOCAL}/model-staging"
        cp -r "${NSS_MODELS_DIR}/${model_id}" "${staging_tmp}"
        # The rename fails if another job on this node finished staging first.
        mv -T "${staging_tmp}" "${staged_model}" 2>/dev/null || rm -rf "${staging_tmp}"
    fi
    if [[ ! -d "${staged_model}" ]]; then
        echo "[NSS SLURM] ERROR: staging ${model_id} to ${staged_model} failed" >&2
        exit 1
    fi
    model_args=(--training__pretrained_model "${staged_model}")
    echo "[NSS SLURM] using staged model ${staged_model}"
elif [[ -n "${model_id}" ]]; then
    echo "[NSS SLURM] ${model_id} not found under ${NSS_MODELS_DIR:-<unset>}; downloading through ${HF_HOME}"
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
echo "[NSS SLURM] job_local=${JOB_LOCAL}"
echo "CUDA_VISIBLE_DEVICES is set to: ${CUDA_VISIBLE_DEVICES:-not set}"

dataset_registry_args=()
dataset_registry_file="${NSS_SHARED_DIR}/dataset_registry.yaml"
if [ -f "${dataset_registry_file}" ]; then
    dataset_registry_args=(--dataset-registry "${dataset_registry_file}")
    echo "Using dataset registry file: ${dataset_registry_file}"
else
    echo "No dataset registry file found at ${dataset_registry_file}"
fi


if [[ "${NSS_PHASE}" == "train" ]]; then
    # Stage 1: PII replacement + training
    # Creates new workdir at run_path with adapter
    "${NSS_RUN_CMD}" run train \
        --data-source "$dataset" \
        --config "$full_config_path" \
        --run-path "$run_path" \
        "${dataset_registry_args[@]}" \
        "${model_args[@]}"
elif [[ "${NSS_PHASE}" == "generate" ]]; then
    # Stage 2: generation (+ optional evaluation)
    # Resumes from existing workdir at run_path
    # Each generation run creates unique output files (synthetic_data_TIMESTAMP.csv)
    # Resume wandb run from train phase if the ID file exists
    wandb_id_file="${run_path}/wandb_run_id.txt"
    wandb_resume_args=()
    if [[ -f "$wandb_id_file" ]]; then
        wandb_resume_args=(--wandb-resume-job-id "$wandb_id_file")
    fi

    "${NSS_RUN_CMD}" run generate \
        --data-source "$dataset" \
        --config "$full_config_path" \
        --run-path "$run_path" \
        "${dataset_registry_args[@]}" \
        "${model_args[@]}" \
        "${wandb_resume_args[@]}"
else
    # Full end-to-end run
    "${NSS_RUN_CMD}" run \
        --data-source "$dataset" \
        --config "$full_config_path" \
        --run-path "$run_path" \
        "${dataset_registry_args[@]}" \
        "${model_args[@]}"
fi

echo "[NSS SLURM] Done"

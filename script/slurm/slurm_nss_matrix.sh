#!/bin/bash

set -euo pipefail

# Source shared environment; prefer existing USER_NAME, then $USER, fallback
USER_NAME="${USER_NAME:-${USER:-seayang}}"

# Source env_variables.sh - use NSS_SLURM_DIR if available, otherwise construct from USER_NAME
if [ -n "${NSS_SLURM_DIR:-}" ]; then
    source "${NSS_SLURM_DIR}/env_variables.sh"
else
    source /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/Safe-Synthesizer/script/slurm/env_variables.sh
fi

# Allow callers to override ADAPTER_PATH after env_variables.sh sets it
if [[ -n "${ADAPTER_PATH_OVERRIDE:-}" ]]; then
    export ADAPTER_PATH="${ADAPTER_PATH_OVERRIDE}"
    export NSS_ARTIFACTS_PATH="${ADAPTER_PATH}"
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

# Temporary: reuse the pre-built venv from the old nmp repo
# (bypasses uv project resolution that fails on missing .nmp_repo)
export VIRTUAL_ENV="${NMP_DIR}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
cd "${NSS_DIR}"

# for column classification
export NIM_ENDPOINT_URL=https://integrate.api.nvidia.com/v1
export NIM_MODEL_ID=qwen/qwen2.5-coder-32b-instruct
source "${LUSTRE_DIR}/.api_tokens.sh"



# Select dataset group
DATASET_GROUP=${DATASET_GROUP:-short}

declare -a datasets

if [ -n "${DATASET_URLS:-}" ]; then
    IFS=',' read -r -a datasets <<< "$DATASET_URLS"
else
    if [ "$DATASET_GROUP" = "long" ]; then
        # 4 longer datasets (require more time)
        datasets=(
            ai_generated_essays
            call_transcripts
            cover_type
            online_news_popularity
            car_accident
            diabetes
        )
    elif [ "$DATASET_GROUP" = "short" ]; then
        # 17 short datasets (21 total minus 4 long)
        datasets=(
            adult
            amazon_reviews_25k
            aids_clinical_trials
            beijing
            bike_sales
            clinc_oos
            default
            dialogs
            EHR
            ecommerce_reviews
            magic
            ontonotes5_reduced
            patient_events
            project_management_sequences
            shoppers
            stack_exchange_data_dump
        )
    fi
fi


RUNS=${RUNS:-5}

# Map index inputs
if [ -n "${DATASET_INDEX:-}" ]; then
    idx=${DATASET_INDEX}
else
    idx=${SLURM_ARRAY_TASK_ID:-0}
fi

num_datasets=${#datasets[@]}
num_runs=$RUNS

if [ "$num_datasets" -eq 0 ]; then
    echo "No datasets defined. Set DATASET_URLS or check DATASET_GROUP." >&2
    exit 1
fi

if [ -n "${RUN_INDEX:-}" ]; then
    # Fixed run index; dataset index comes directly from idx
    run_idx=${RUN_INDEX}
    dataset_idx=$(( idx % num_datasets ))
else
    # Legacy mapping across (dataset, config, run)
    per_dataset=$(( num_configs * num_runs ))
    dataset_idx=$(( (idx / per_dataset) % num_datasets ))
    remainder=$(( idx % per_dataset ))
    run_idx=$(( remainder % num_runs ))
fi

dataset=${datasets[$dataset_idx]}

# Extract dataset name for path construction (handles both full paths and simple names)
# e.g., "/path/to/adult.csv" -> "adult", "/path/to/data.parquet" -> "data", "adult" -> "adult"
# Supports: .csv, .parquet, .json, .jsonl
dataset_basename=$(basename "$dataset")
dataset_name="${dataset_basename%.*}"

config=${CONFIG_DIR}/${CONFIG_NAMES}

# Construct run path for WorkdirStructure
# - two_stage mode (train/generate): Share same base path without SLURM ID
#   Note: Don't submit multiple batches of same config/dataset/run_idx concurrently in two_stage mode
#   The train phase creates the workdir, generate phase resumes from it (with unique output files)
# - end_to_end mode: Include SLURM ID for uniqueness across concurrent runs
slurm_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    slurm_id="${slurm_id}_${SLURM_ARRAY_TASK_ID}"
fi

if [[ "${PHASE}" == "end_to_end" ]]; then
    # end_to_end: include SLURM ID for unique paths across concurrent runs
    run_path=${ADAPTER_PATH}/${CONFIG_NAMES}_${dataset_name}_${run_idx}_${slurm_id}
else
    # two_stage (train/generate): shared base path without SLURM ID
    # Train creates workdir; generate resumes and creates unique output files per run
    run_path=${ADAPTER_PATH}/${CONFIG_NAMES}_${dataset_name}_${run_idx}
fi

# Results directory (optional; experiments may also write their own outputs)
# EXP_NAME controls the experiment namespace under nss_results
EXP_NAME=${EXP_NAME:-multi_jobs}
OUTPUT_DIR="${BASE_LOG_DIR}/${EXP_NAME}/$DATASET_GROUP/${dataset_name}/$(basename "$config" .yaml)/run_${run_idx}"
# mkdir -p "$OUTPUT_DIR"

# Derive a stable seed unless overridden
SEED=${SEED_OVERRIDE:-$(( 42 + dataset_idx*100 +  run_idx ))}
export SEED
export RUN_INDEX=$run_idx
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
echo "[Matrix] idx=$idx group=$DATASET_GROUP dataset_idx=$dataset_idx run_idx=$run_idx"
echo "[Matrix] dataset=$dataset (name=$dataset_name) config=$config runs=$RUNS seed=$SEED"
echo "[Matrix] run_path=$run_path"
echo "[Matrix] output_dir=$OUTPUT_DIR"

dataset_registry_arg=""
dataset_registry_file="${NSS_SHARED_DIR}/dataset_registry.yaml"
if [ -f "${dataset_registry_file}" ]; then
    dataset_registry_arg="--dataset-registry ${dataset_registry_file}"
    echo "Using dataset registry file: ${dataset_registry_file}"
else
    echo "No dataset registry file found at ${dataset_registry_file}"
fi

# Detect separate test dataset (convention: *_train_nss.csv → *_test_nss.csv)
test_url_arg=""
test_dataset="${dataset/_train_nss/_test_nss}"
if [[ "$test_dataset" != "$dataset" ]] && [[ -f "$test_dataset" ]]; then
    test_url_arg="--test-url $test_dataset"
    echo "[Matrix] Using separate test dataset: $test_dataset"
fi

# Auto-scale num_input_records_to_sample based on training data size.
# Only applies when the dataset is a local CSV file (full path).
# Since `run train` doesn't support pydantic override flags, we create a
# temporary copy of the config YAML with the value patched in.
effective_config="${config}.yaml"
if [[ "$dataset" == *.csv ]] && [[ -f "$dataset" ]]; then
    # wc -l counts lines; subtract 1 for the header row
    line_count=$(wc -l < "$dataset")
    num_rows=$(( line_count > 0 ? line_count - 1 : 0 ))
    if [[ $num_rows -gt 0 ]]; then
        if [[ $num_rows -lt 10000 ]]; then
            multiplier=15
        elif [[ $num_rows -lt 50000 ]]; then
            multiplier=10
        elif [[ $num_rows -lt 100000 ]]; then
            multiplier=3
        else
            multiplier=2
        fi
        scaled_records=$(( num_rows * multiplier ))
        echo "[Matrix] Auto-scaled num_input_records_to_sample: ${num_rows} rows x ${multiplier} = ${scaled_records}"

        # Create a temp config with the overridden value
        tmp_config=$(mktemp /tmp/nss_config_XXXXXX.yaml)
        sed "s/^\(\s*num_input_records_to_sample:\s*\).*/\1${scaled_records}/" "${config}.yaml" > "$tmp_config"
        effective_config="$tmp_config"
        echo "[Matrix] Using temp config: $effective_config"
    fi
fi

PHASE=${PHASE:-end_to_end}
if [[ "$PHASE" == "train" ]]; then
    # Stage 1: PII replacement + training
    # Creates new workdir at run_path with adapter
    safe-synthesizer run train \
        --url "$dataset" \
        --config "$effective_config" \
        --run-path "$run_path" \
        $dataset_registry_arg \
        $test_url_arg
elif [[ "$PHASE" == "generate" ]]; then
    # Stage 2: generation (+ optional evaluation)
    # Resumes from existing workdir at run_path
    # Each generation run creates unique output files (synthetic_data_TIMESTAMP.csv)
    # Resume wandb run from train phase if the ID file exists
    wandb_id_file="${run_path}/wandb_run_id.txt"
    wandb_resume_arg=""
    if [[ -f "$wandb_id_file" ]]; then
        wandb_resume_arg="--wandb-resume-job-id $wandb_id_file"
    fi

    safe-synthesizer run generate \
        --url "$dataset" \
        --config "$effective_config" \
        --run-path "$run_path" \
        $dataset_registry_arg \
        $wandb_resume_arg
else
    # Full end-to-end run
    safe-synthesizer run \
        --url "$dataset" \
        --config "$effective_config" \
        --run-path "$run_path" \
        $dataset_registry_arg \
        $test_url_arg
fi
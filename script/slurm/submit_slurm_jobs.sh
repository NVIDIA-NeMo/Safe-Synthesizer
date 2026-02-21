#!/bin/bash

# Submit script for running NSS experiments on a slurm cluster.

set -euo pipefail

# Defaults; override with flags below.
RUNS="1"
PARTITION="polar4"
EXP_NAME="nss_exp"
DATASET_GROUP=""
DATASET_URLS_CSV=""
PIPELINE_MODE="two_stage"  # values: two_stage | end_to_end
CONFIGS_CSV=""  # optional override for CONFIGS array (comma-separated)
WANDB_PROJECT="" # optional wandb project name; uses EXP_NAME if not provided
MAX_CONCURRENT_SLURM_JOBS="" # optional max number of concurrent slurm jobs to run within each array; if not provided, no restriction is applied
ACCOUNT="${ACCOUNT:-llmservice_sdg_research}"
TIME_LIMIT="04:00:00"
TRAIN_TIME_LIMIT=""
GENERATE_TIME_LIMIT=""

# Parse flags (order-independent). Unknown flags are ignored.
while [ $# -gt 0 ]; do
  case "$1" in
    --configs|-c)
      CONFIGS_CSV="${2:-}"; shift 2;;
    --dataset-urls|-d)
      DATASET_URLS_CSV="${2:-}"; shift 2;;
    --runs|-r)
      RUNS="${2:-$RUNS}"; shift 2;;
    --partition|-p)
      PARTITION="${2:-$PARTITION}"; shift 2;;
    --exp-name|-e)
      EXP_NAME="${2:-$EXP_NAME}"; shift 2;;
    --dataset-group|-g)
      DATASET_GROUP="${2:-$DATASET_GROUP}"; shift 2;;
    --pipeline-mode|-m)
      PIPELINE_MODE="${2:-$PIPELINE_MODE}"; shift 2;;
    --wandb-project|-w)
      WANDB_PROJECT="${2:-}"; shift 2;;
    --max-concurrent-slurm-jobs)
      MAX_CONCURRENT_SLURM_JOBS="${2:-$MAX_CONCURRENT_SLURM_JOBS}"; shift 2;;
    --time-limit|-t)
      TIME_LIMIT="${2:-$TIME_LIMIT}"; shift 2;;
    --train-time-limit)
      TRAIN_TIME_LIMIT="${2:-$TRAIN_TIME_LIMIT}"; shift 2;;
    --generate-time-limit)
      GENERATE_TIME_LIMIT="${2:-$GENERATE_TIME_LIMIT}"; shift 2;;
    --dry-run)
      DRY_RUN="true"; shift;;
    --help|-h)
      echo "Usage: $0 [--configs c1,c2] [--dataset-urls name1,url1,path1] [--dataset-group short|long] [--runs N] [--exp-name NAME] [--pipeline-mode two_stage|end_to_end] [--partition P] [--wandb-project PROJECT] [--max-concurrent-slurm-jobs N] [--time-limit TIME] [--train-time-limit TIME] [--generate-time-limit TIME] [--dry-run]"
      echo ""
      echo "Provide either --dataset-urls to specify a list of datasets by name, url, or path, or --dataset-group to use a predefined set of datasets."
      echo "Time limits:"
      echo "    --time-limit is used for end_to_end mode (defaults to 4 hours)"
      echo "    --train-time-limit and --generate-time-limit are used for two_stage mode, and will default to --time-limit the more more specific train and generate limits are not provided"

      exit 0;;
    --) shift; break;;
    *) shift;;  # ignore unknown
  esac
done

if [[ (-n "${DATASET_URLS_CSV:-}" && -n "${DATASET_GROUP:-}") || \
      (-z "${DATASET_URLS_CSV:-}" && -z "${DATASET_GROUP:-}") ]]; then
  echo "ERROR: Exactly one of --dataset-urls and --dataset-group must be provided." >&2
  echo " DATASET_URLS: ${DATASET_URLS_CSV:-}" >&2
  echo " DATASET_GROUP: ${DATASET_GROUP:-}" >&2
  exit 1
fi


if [[ -z "${USER_NAME:-}" ]]; then
  echo "ERROR: USER_NAME is not set. Please export it before submitting." >&2
  echo "Example: export USER_NAME=your_lustre_username" >&2
  exit 1
fi

if [[ -z "${TRAIN_TIME_LIMIT:-}" ]]; then
  TRAIN_TIME_LIMIT="${TIME_LIMIT}"
fi
if [[ -z "${GENERATE_TIME_LIMIT:-}" ]]; then
  GENERATE_TIME_LIMIT="${TIME_LIMIT}"
fi

source env_variables.sh

# Idle GPU exemption comment (15 minutes for data loading/preprocessing)
IDLE_EXEMPT_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"15","reason":"data_loading","description":"Data loading and preprocessing grace period for this job"}}'

# Verify token file exists for NIM usage inside jobs
if [[ ! -f "${LUSTRE_DIR}/.api_tokens.sh" ]]; then
  echo "ERROR: ${LUSTRE_DIR}/.api_tokens.sh not found." >&2
  echo "Create it and define NIM_API_KEY." >&2
  exit 1
fi

# Build configs list: CLI override (comma-separated) takes precedence; otherwise use CONFIGS from env
declare -a CONFIGS_LIST
if [ -n "${CONFIGS_CSV:-}" ]; then
  IFS=',' read -r -a CONFIGS_LIST <<< "${CONFIGS_CSV}"
else
  CONFIGS_LIST=("${CONFIGS[@]}")
fi

EXP_LOG_DIR=${BASE_LOG_DIR}/${EXP_NAME}

# Set WANDB_PROJECT to custom value or default to EXP_NAME
export WANDB_PROJECT="${WANDB_PROJECT:-$EXP_NAME}"

echo "Submitting array jobs (pipeline=${PIPELINE_MODE}, RUNS=${RUNS}, PARTITION=${PARTITION}, EXP_NAME=${EXP_NAME}, GROUP=${DATASET_GROUP}, WANDB_PROJECT=${WANDB_PROJECT})"

# Array variable of datasets to process
declare -a DATASETS

if [[ -n "${DATASET_URLS_CSV:-}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASET_URLS_CSV}"
elif [[ "${DATASET_GROUP:-}" == "long" ]]; then
  DATASETS=(
    ai_generated_essays
    call_transcripts
    cover_type
    online_news_popularity
    car_accident
    diabetes
  )
elif [[ "${DATASET_GROUP:-}" == "short" ]]; then
  DATASETS=(
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
else
  echo "ERROR: Unrecognized dataset group: ${DATASET_GROUP}" >&2
  exit 1
fi

num_datasets=${#DATASETS[@]}

echo "DATASETS: ${DATASETS[*]}"
echo "CONFIGS_LIST: ${CONFIGS_LIST[*]}"
num_configs=${#CONFIGS_LIST[@]}

total_tasks=$(( num_datasets * num_configs * RUNS ))

echo "total_tasks: ${total_tasks}"

# Assumes dataset and config names do not contain commas

# Use a single job array to submit all experiments across datasets and configs
# (Except for two_stage, we'll have 2 job arrays: one for train and one for generate)
# Pack the dataset and config names for the array into 2 comma-separated strings
# defining all the experiments to execute.
PACKED_CONFIGS=""
PACKED_DATASETS=""
# Probably a more efficient way to do this, but this is simple and works.
for config in "${CONFIGS_LIST[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for run_idx in $(seq 0 $((RUNS-1))); do
      PACKED_CONFIGS="${PACKED_CONFIGS:+$PACKED_CONFIGS,}${config}"
      PACKED_DATASETS="${PACKED_DATASETS:+$PACKED_DATASETS,}${dataset}"
    done
  done
done
export PACKED_DATASETS
export PACKED_CONFIGS

echo "PACKED_DATASETS: ${PACKED_DATASETS}"
echo "PACKED_CONFIGS: ${PACKED_CONFIGS}"

# To reduce startup time and avoid timeouts downloading the image, use an existing
# image already saved on lustre if possible.
CACHED_IMAGE_PATH="${NSS_SHARED_DIR}/images/cuda_12_8_1_cudnn_runtime_ubuntu24_04.sqsh"
if [ -f "${CACHED_IMAGE_PATH}" ]; then
  CONTAINER_IMAGE="${CACHED_IMAGE_PATH}"
else
  CONTAINER_IMAGE="nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
fi
echo "Using container image: ${CONTAINER_IMAGE}"
CONTAINER_MOUNTS="/lustre:/lustre"

export CONTAINER_IMAGE
export CONTAINER_MOUNTS
export ACCOUNT

mkdir -p "${EXP_LOG_DIR}"

# No customized time limits for the moment, we can add them back later if we
# see them providing value. Given variability in expected runtime for configs
# x dataset, unclear if it's worth it.


# Setup array args for sbatch
array_spec="0-$((total_tasks-1))"
if [ -n "${MAX_CONCURRENT_SLURM_JOBS:-}" ]; then
  echo "Restricted to at most ${MAX_CONCURRENT_SLURM_JOBS} concurrent jobs with each array."
  if [[ "${PIPELINE_MODE}" == "two_stage" ]]; then
    echo "NOTE: With --pipeline-mode two_stage, we will submit two arrays, so the maximum potential concurrent jobs will be ${MAX_CONCURRENT_SLURM_JOBS} * 2. But due to generate needing to run after train, mostly we should stay well below that 2x limit."
  fi
  array_spec="${array_spec}%${MAX_CONCURRENT_SLURM_JOBS}"
fi

echo "array_spec: ${array_spec}"


common_args=(
  --parsable
  --partition "${PARTITION}"
  --account llmservice_sdg_research
  --comment "${IDLE_EXEMPT_COMMENT}"
  --output ${EXP_LOG_DIR}/slurm_%A_%a.out
  --error ${EXP_LOG_DIR}/slurm_%A_%a.err
  --nodes 1
  --ntasks-per-node 1
  --cpus-per-task 16
  --mem 32G
  --gres gpu:1
  --array "${array_spec}"
)



if [ -n "${DRY_RUN:-}" ]; then
  echo "--dry-run enabled, no jobs will be submitted, using --test-only flag with sbatch"
  common_args+=("--test-only")
fi

if [[ "${PIPELINE_MODE}" == "two_stage" ]]; then
  # Submit two job arrays with sbatch, one for train and one for generate. Uses
  # --dependency=aftercorr to ensure generate jobs only run after train jobs
  # have completed successfully.

  train_array_id=$( \
    sbatch "${common_args[@]}" \
      --job-name nss_train \
      --export=ALL,NSS_PHASE=train,TIME_LIMIT=${TRAIN_TIME_LIMIT} \
      ${NSS_SLURM_DIR}/slurm_srun.sh )

  gen_array_id=$( \
    sbatch "${common_args[@]}" \
      --job-name nss_generate \
      --dependency=aftercorr:${train_array_id} \
      --export=ALL,NSS_PHASE=generate,TIME_LIMIT=${GENERATE_TIME_LIMIT} \
      ${NSS_SLURM_DIR}/slurm_srun.sh )

else
  sbatch "${common_args[@]}" \
    --job-name nss_end_to_end \
    --export=ALL,NSS_PHASE=end_to_end,TIME_LIMIT=${TIME_LIMIT} \
    ${NSS_SLURM_DIR}/slurm_srun.sh
fi

echo "Done submitting jobs to slurm cluster. Monitor with: squeue -u ${USER_NAME}"

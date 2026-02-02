#!/bin/bash

set -euo pipefail

# Matrix submitter across built-in dataset groups in slurm_nss_matrix.sh
# Usage:
#   bash exp/submit_slurm_jobs.sh [CONFIGS_CSV] [RUNS] [PARTITION] [EXP_NAME] [DATASET_GROUP] [SLEEP_SEC] [PIPELINE_MODE]
# Examples:
#   bash exp/submit_slurm_jobs.sh unsloth,dp,dp_usg_guidance 5 polar4 matrix_exp short 5 two_stage

# Defaults; override with flags below. CONFIGS now come from env_variables.sh (CONFIGS array)
RUNS="5"
PARTITION="polar4"
EXP_NAME="matrix_exp"
DATASET_GROUP="short"
SLEEP_SEC="5"
PIPELINE_MODE="two_stage"  # values: two_stage | end_to_end
CONFIGS_CSV=""  # optional override for CONFIGS array (comma-separated)
SUBMIT_MODE="array"  # values: array | sequential
WANDB_PROJECT="" # optional wandb project name; uses EXP_NAME if not provided

# Parse flags (order-independent). Unknown flags are ignored.
while [ $# -gt 0 ]; do
  case "$1" in
    --configs|-c)
      CONFIGS_CSV="${2:-}"; shift 2;;
    --submit-mode)
      SUBMIT_MODE="${2:-$SUBMIT_MODE}"; shift 2;;
    --runs|-r)
      RUNS="${2:-$RUNS}"; shift 2;;
    --partition|-p)
      PARTITION="${2:-$PARTITION}"; shift 2;;
    --exp-name|-e)
      EXP_NAME="${2:-$EXP_NAME}"; shift 2;;
    --dataset-group|-g)
      DATASET_GROUP="${2:-$DATASET_GROUP}"; shift 2;;
    --sleep-sec|-s)
      SLEEP_SEC="${2:-$SLEEP_SEC}"; shift 2;;
    --pipeline-mode|-m)
      PIPELINE_MODE="${2:-$PIPELINE_MODE}"; shift 2;;
    --wandb-project|-w)
      WANDB_PROJECT="${2:-}"; shift 2;;
    --help|-h)
      echo "Usage: $0 [--configs c1,c2] [--runs N] [--partition P] [--exp-name NAME] [--dataset-group short|long] [--sleep-sec S] [--pipeline-mode two_stage|end_to_end] [--submit-mode array|sequential] [--wandb-project PROJECT]"
      exit 0;;
    --) shift; break;;
    *) shift;;  # ignore unknown
  esac
done

if [[ -z "${USER_NAME:-}" ]]; then
  echo "ERROR: USER_NAME is not set. Please export it before submitting." >&2
  echo "Example: export USER_NAME=your_lustre_username" >&2
  exit 1
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
if [ -n "${CONFIGS_CSV:-}" ]; then
  IFS=',' read -r -a CONFIGS_LIST <<< "${CONFIGS_CSV}"
else
  CONFIGS_LIST=("${CONFIGS[@]}")
fi

EXP_LOG_DIR=${BASE_LOG_DIR}/${EXP_NAME}
mkdir -p "${EXP_LOG_DIR}"

# Set WANDB_PROJECT to custom value or default to EXP_NAME
export WANDB_PROJECT="${WANDB_PROJECT:-$EXP_NAME}"

echo "Submitting matrix jobs (pipeline=${PIPELINE_MODE}, RUNS=${RUNS}, PARTITION=${PARTITION}, EXP_NAME=${EXP_NAME}, GROUP=${DATASET_GROUP}, WANDB_PROJECT=${WANDB_PROJECT})"

# Mirror dataset counts used in slurm_nss_matrix.sh to size the array correctly
declare -a DATASETS
if [[ "${DATASET_GROUP}" == "long" ]]; then
  DATASETS=(
    ai_generated_essays
    call_transcripts
    cover_type
    online_news_popularity
    car_accident
    diabetes
  )
else
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
fi
num_datasets=${#DATASETS[@]}

total_tasks=$(( num_datasets * RUNS ))

for config in "${CONFIGS_LIST[@]}"; do
  mkdir -p "${EXP_LOG_DIR}/${DATASET_GROUP}/${config}"

  # Resolve time limit similarly to submit_single_dataset.sh
  base_key="${config}"
  if [[ "${config}" == *unsloth* ]]; then
    base_key="unsloth"
  elif [[ "${config}" == *dp* ]]; then
    base_key="dp"
  else
    base_key="max"
  fi
  if [[ "${DATASET_GROUP}" == "long" ]]; then
    TIME_LIMIT="${CONFIG_TIME_LIMITS_LONG[${base_key}]}"
  else
    TIME_LIMIT="${CONFIG_TIME_LIMITS_SHORT[${base_key}]}"
  fi

  # Derive per-phase time limits with safe fallbacks to the general limit
  if [[ "${DATASET_GROUP}" == "long" ]]; then
    default_time="${CONFIG_TIME_LIMITS_LONG[${base_key}]}"
    if declare -p CONFIG_TRAIN_TIME_LIMITS_LONG >/dev/null 2>&1; then
      train_time_candidate="${CONFIG_TRAIN_TIME_LIMITS_LONG[${base_key}]:-}"
    else
      train_time_candidate=""
    fi
    if declare -p CONFIG_GEN_TIME_LIMITS_LONG >/dev/null 2>&1; then
      gen_time_candidate="${CONFIG_GEN_TIME_LIMITS_LONG[${base_key}]:-}"
    else
      gen_time_candidate=""
    fi
  else
    default_time="${CONFIG_TIME_LIMITS_SHORT[${base_key}]}"
    if declare -p CONFIG_TRAIN_TIME_LIMITS_SHORT >/dev/null 2>&1; then
      train_time_candidate="${CONFIG_TRAIN_TIME_LIMITS_SHORT[${base_key}]:-}"
    else
      train_time_candidate=""
    fi
    if declare -p CONFIG_GEN_TIME_LIMITS_SHORT >/dev/null 2>&1; then
      gen_time_candidate="${CONFIG_GEN_TIME_LIMITS_SHORT[${base_key}]:-}"
    else
      gen_time_candidate=""
    fi
  fi
  TIME_LIMIT_TRAIN="${train_time_candidate:-$default_time}"
  TIME_LIMIT_GEN="${gen_time_candidate:-$default_time}"

  if [[ "${PIPELINE_MODE}" == "two_stage" ]]; then
    if [[ "${SUBMIT_MODE}" == "sequential" ]]; then
      echo "[${config}] Submitting sequential by run-index (all datasets for run 0, then run 1, ...). TRAIN→GEN uses afterok; cross-run uses afterany on prior GEN."
      # Track previous run's GEN job id per dataset index
      declare -a PREV_GEN_JOB_IDS
      for (( run_idx=0; run_idx<${RUNS}; run_idx++ )); do
        # First submit all TRAIN jobs for this run index
        declare -a TRAIN_JOB_IDS
        for (( d=0; d<${num_datasets}; d++ )); do
          dep_args=()
          if [[ -n "${PREV_GEN_JOB_IDS[d]:-}" ]]; then
            dep_args+=(--dependency=afterany:${PREV_GEN_JOB_IDS[d]})
          fi
          train_job_id=$( \
            sbatch --parsable \
              --partition=${PARTITION} \
              --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_d${d}_r${run_idx}_train \
              --account=llmservice_sdg_research \
              --comment="${IDLE_EXEMPT_COMMENT}" \
              --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_train_%j.out \
              --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_train_%j.err \
              --time=${TIME_LIMIT_TRAIN} \
              --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
              ${dep_args:+${dep_args[@]}} \
              --export=ALL,PHASE=train,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_TRAIN},DATASET_INDEX=${d},RUN_INDEX=${run_idx},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
              ${NSS_SLURM_DIR}/slurm_srun.sh )
          TRAIN_JOB_IDS[d]="${train_job_id}"
          echo "[${config}] dataset_idx=${d} run_index=${run_idx} TRAIN submitted ${train_job_id}"
        done
        # Then submit all GEN jobs for this run index with dependency on corresponding TRAIN
        for (( d=0; d<${num_datasets}; d++ )); do
          gen_job_id=$( \
            sbatch --parsable \
              --partition=${PARTITION} \
              --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_d${d}_r${run_idx}_gen \
              --account=llmservice_sdg_research \
              --comment="${IDLE_EXEMPT_COMMENT}" \
              --dependency=afterok:${TRAIN_JOB_IDS[d]} \
              --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_gen_%j.out \
              --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_gen_%j.err \
              --time=${TIME_LIMIT_GEN} \
              --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
              --export=ALL,PHASE=generate,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_GEN},DATASET_INDEX=${d},RUN_INDEX=${run_idx},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
              ${NSS_SLURM_DIR}/slurm_srun.sh )
          PREV_GEN_JOB_IDS[d]="${gen_job_id}"
          echo "[${config}] dataset_idx=${d} run_index=${run_idx} GEN submitted ${gen_job_id} (afterok ${TRAIN_JOB_IDS[d]})"
        done
        echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
      done
    else
      echo "[${config}] Submitting TRAIN array of ${total_tasks} tasks (runs=${RUNS}, datasets=${num_datasets}, time=${TIME_LIMIT_TRAIN})"
      train_array_id=$( \
        sbatch --parsable \
          --partition=${PARTITION} \
          --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_train \
          --account=llmservice_sdg_research \
          --comment="${IDLE_EXEMPT_COMMENT}" \
          --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_train_%A_%a.out \
          --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_train_%A_%a.err \
          --time=${TIME_LIMIT_TRAIN} \
          --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
          --array=0-$((total_tasks-1)) \
          --export=ALL,PHASE=train,ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_TRAIN},num_configs=1 \
          ${NSS_SLURM_DIR}/slurm_srun.sh )

      echo "[${config}] Submitting GEN array dependent on TRAIN (aftercorr ${train_array_id}, time=${TIME_LIMIT_GEN})"
      gen_array_id=$( \
        sbatch --parsable \
          --partition=${PARTITION} \
          --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_gen \
          --account=llmservice_sdg_research \
          --comment="${IDLE_EXEMPT_COMMENT}" \
          --dependency=aftercorr:${train_array_id} \
          --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_gen_%A_%a.out \
          --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_gen_%A_%a.err \
          --time=${TIME_LIMIT_GEN} \
          --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
          --array=0-$((total_tasks-1)) \
          --export=ALL,PHASE=generate,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_GEN},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
          ${NSS_SLURM_DIR}/slurm_srun.sh )

      echo "[${config}] TRAIN ${train_array_id}, GEN ${gen_array_id}"
      echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
    fi
  else
    if [[ "${SUBMIT_MODE}" == "sequential" ]]; then
      echo "[${config}] Submitting end-to-end sequential by run-index (all datasets for run 0, then run 1, ...). Each run waits on prior run's job per dataset (afterany)."
      declare -a PREV_E2E_JOB_IDS
      for (( run_idx=0; run_idx<${RUNS}; run_idx++ )); do
        for (( d=0; d<${num_datasets}; d++ )); do
          dep_args=()
          if [[ -n "${PREV_E2E_JOB_IDS[d]:-}" ]]; then
            dep_args+=(--dependency=afterany:${PREV_E2E_JOB_IDS[d]})
          fi
          job_id=$( \
            sbatch --parsable \
              --partition=${PARTITION} \
              --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_d${d}_r${run_idx}_single \
              --account=llmservice_sdg_research \
              --comment="${IDLE_EXEMPT_COMMENT}" \
              --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_%j.out \
              --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_%j.err \
              --time=${TIME_LIMIT} \
              --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
              ${dep_args:+${dep_args[@]}} \
              --export=ALL,PHASE=end_to_end,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT},DATASET_INDEX=${d},RUN_INDEX=${run_idx},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
              ${NSS_SLURM_DIR}/slurm_srun.sh )
          PREV_E2E_JOB_IDS[d]="${job_id}"
          echo "[${config}] dataset_idx=${d} run_index=${run_idx} submitted job ${job_id}"
        done
        echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
      done
    else
      echo "[${config}] Submitting end-to-end array of ${total_tasks} tasks"
      job_id=$( \
        sbatch --parsable \
          --partition=${PARTITION} \
          --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_single \
          --account=llmservice_sdg_research \
          --comment="${IDLE_EXEMPT_COMMENT}" \
          --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_%A_%a.out \
          --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_%A_%a.err \
          --time=${TIME_LIMIT} \
          --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
          --array=0-$((total_tasks-1)) \
          --export=ALL,PHASE=end_to_end,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
          ${NSS_SLURM_DIR}/slurm_srun.sh )
      echo "[${config}] submitted job ${job_id}"
      echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
    fi
  fi
done

echo "Done. Monitor with: squeue -u ${USER_NAME}"

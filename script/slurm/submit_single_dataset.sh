#!/bin/bash

set -euo pipefail

# One-off dataset submitter.
# Usage:
#   bash exp/submit_single_dataset.sh <DATASET_PATH_OR_URL> [CONFIGS_CSV] [RUNS] [PARTITION] [EXP_NAME] [DATASET_GROUP] [SLEEP_SEC] [SUBMIT_MODE]
#
# SUBMIT_MODE:
#   - array (default): submit a SLURM array over run indices per config
#   - sequential: submit individual jobs per run index per config, chaining with dependency within the same config (run k depends on run k-1 completion, regardless of success)
#
# Examples:
#   bash exp/submit_single_dataset.sh \
#     /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/safe-synthetics/cleaned/Adobe-2k.csv \
#     unsloth,dp,dp_usg_guidance 5 polar4 regex_adobe2k short 5 two_stage

# Defaults; override with flags below. CONFIGS now come from env_variables.sh (CONFIGS array)
DATASET_URLS=""
RUNS="5"
PARTITION="polar4"
EXP_NAME="single_ds"
DATASET_GROUP="short"
SLEEP_SEC="5"
SUBMIT_MODE="array"      # values: array | sequential
PIPELINE_MODE="two_stage"  # values: two_stage | end_to_end
CONFIGS_CSV=""  # optional override for CONFIGS array (comma-separated)
WANDB_PROJECT="" # optional wandb project name; uses EXP_NAME if not provided

# Parse flags (order-independent). Unknown flags are ignored.
while [ $# -gt 0 ]; do
  case "$1" in
    --configs|-c)
      CONFIGS_CSV="${2:-}"; shift 2;;
    --dataset-urls|-d)
      DATASET_URLS="${2:-$DATASET_URLS}"; shift 2;;
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
    --submit-mode)
      SUBMIT_MODE="${2:-$SUBMIT_MODE}"; shift 2;;
    --pipeline-mode|-m)
      PIPELINE_MODE="${2:-$PIPELINE_MODE}"; shift 2;;
    --wandb-project|-w)
      WANDB_PROJECT="${2:-}"; shift 2;;
    --help|-h)
      echo "Usage: $0 --dataset-urls PATH_OR_URL [--configs c1,c2] [--runs N] [--partition P] [--exp-name NAME] [--dataset-group short|long] [--sleep-sec S] [--submit-mode array|sequential] [--pipeline-mode two_stage|end_to_end] [--wandb-project PROJECT]"
      exit 0;;
    --) shift; break;;
    *)
      # Allow first bare arg to be dataset URLs for convenience
      if [ -z "$DATASET_URLS" ]; then
        DATASET_URLS="$1"; shift
      else
        shift
      fi
      ;;
  esac
done

if [[ -z "${DATASET_URLS}" ]]; then
  echo "ERROR: DATASET_URLS (arg1) is required (path or URL to CSV)." >&2
  exit 1
fi

# Require USER_NAME to be explicitly set before submitting jobs
if [[ -z "${USER_NAME:-}" ]]; then
  echo "ERROR: USER_NAME is not set. Please export it before submitting." >&2
  echo "Example: export USER_NAME=your_lustre_username" >&2
  exit 1
fi

source env_variables.sh

# Idle GPU exemption comment (15 minutes for data loading/preprocessing)
IDLE_EXEMPT_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"15","reason":"data_loading","description":"Data loading and preprocessing grace period for this job"}}'

# Verify API token file exists and instruct to add NIM_API_KEY if missing
if [[ ! -f "${LUSTRE_DIR}/.api_tokens.sh" ]]; then
  echo "ERROR: ${LUSTRE_DIR}/.api_tokens.sh not found." >&2
  echo "Create it and define NIM_API_KEY, e.g.:" >&2
  echo "  echo 'export NIM_API_KEY=\"<your_api_key>\"' > ${LUSTRE_DIR}/.api_tokens.sh" >&2
  echo "  chmod 600 ${LUSTRE_DIR}/.api_tokens.sh" >&2
  exit 1
fi

# Source tokens and ensure NIM_API_KEY is present
source "${LUSTRE_DIR}/.api_tokens.sh"
if [[ -z "${NIM_API_KEY:-}" ]]; then
  echo "ERROR: NIM_API_KEY is not set in ${LUSTRE_DIR}/.api_tokens.sh." >&2
  echo "Edit the file and add: export NIM_API_KEY=\"<your_api_key>\"" >&2
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

echo "Submitting single-dataset jobs (mode=${SUBMIT_MODE}, pipeline=${PIPELINE_MODE}, RUNS=${RUNS}, PARTITION=${PARTITION}, EXP_NAME=${EXP_NAME}, DATASET_GROUP=${DATASET_GROUP}, WANDB_PROJECT=${WANDB_PROJECT})"
echo "Dataset(s): ${DATASET_URLS}"
echo "Configs: ${CONFIGS_LIST[*]}"

for config in "${CONFIGS_LIST[@]}"; do
  mkdir -p "${EXP_LOG_DIR}/${DATASET_GROUP}/${config}"

  # Resolve dynamic time limits by pattern (unsloth > dp > exact key)
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
      echo "[${config}] Submitting ${RUNS} TRAIN→GEN sequential runs with dependency afterok (train=${TIME_LIMIT_TRAIN}, gen=${TIME_LIMIT_GEN})"
      for run_idx in $(seq 0 $((RUNS-1))); do
        train_job_id=$( \
          sbatch --parsable \
            --partition=${PARTITION} \
            --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_r${run_idx}_train \
            --account=llmservice_sdg_research \
            --comment="${IDLE_EXEMPT_COMMENT}" \
            --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_train_%j.out \
            --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_train_%j.err \
            --time=${TIME_LIMIT_TRAIN} \
            --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
            --export=ALL,PHASE=train,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_TRAIN},DATASET_URLS=${DATASET_URLS},RUN_INDEX=${run_idx},DATASET_INDEX=0,num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
            ${NSS_SLURM_DIR}/slurm_srun.sh )
        echo "[${config}] run_index=${run_idx} TRAIN submitted ${train_job_id}"

        gen_job_id=$( \
          sbatch --parsable \
            --partition=${PARTITION} \
            --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_r${run_idx}_gen \
            --account=llmservice_sdg_research \
            --comment="${IDLE_EXEMPT_COMMENT}" \
            --dependency=afterok:${train_job_id} \
            --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_gen_%j.out \
            --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_gen_%j.err \
            --time=${TIME_LIMIT_GEN} \
            --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
            --export=ALL,PHASE=generate,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_GEN},DATASET_URLS=${DATASET_URLS},RUN_INDEX=${run_idx},DATASET_INDEX=0,num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
            ${NSS_SLURM_DIR}/slurm_srun.sh )
        echo "[${config}] run_index=${run_idx} GEN submitted ${gen_job_id} (afterok ${train_job_id})"
        echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
      done
    else
      echo "[${config}] Submitting TRAIN array of ${RUNS} (time=${TIME_LIMIT_TRAIN})"
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
          --array=0-$((RUNS-1)) \
          --export=ALL,PHASE=train,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_TRAIN},DATASET_URLS=${DATASET_URLS},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
          ${NSS_SLURM_DIR}/slurm_srun.sh )

      echo "[${config}] Submitting GEN array of ${RUNS} dependent on TRAIN (aftercorr ${train_array_id}, time=${TIME_LIMIT_GEN})"
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
          --array=0-$((RUNS-1)) \
          --export=ALL,PHASE=generate,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT_GEN},DATASET_URLS=${DATASET_URLS},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
          ${NSS_SLURM_DIR}/slurm_srun.sh )

      echo "[${config}] TRAIN ${train_array_id}, GEN ${gen_array_id}"
      echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
    fi
  else
    # Backward-compatible: single end-to-end job(s)
    if [[ "${SUBMIT_MODE}" == "sequential" ]]; then
      echo "[${config}] Submitting ${RUNS} sequential runs with per-config afterany dependency (time=${TIME_LIMIT})"
      prev_job_id=""
      for run_idx in $(seq 0 $((RUNS-1))); do
        dep_args=()
        if [[ -n "${prev_job_id}" ]]; then
          dep_args+=(--dependency=afterany:${prev_job_id})
        fi
        job_id=$( \
          sbatch --parsable \
            --partition=${PARTITION} \
            --job-name=${USER_NAME}_nss_${DATASET_GROUP}_${config}_r${run_idx}_single \
            --account=llmservice_sdg_research \
            --comment="${IDLE_EXEMPT_COMMENT}" \
            --output=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_%j.out \
            --error=${EXP_LOG_DIR}/${DATASET_GROUP}/${config}/slurm_%j.err \
            --time=${TIME_LIMIT} \
            --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=32G --gres=gpu:1 \
            ${dep_args:+${dep_args[@]}} \
            --export=ALL,PHASE=end_to_end,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT},DATASET_URLS=${DATASET_URLS},RUN_INDEX=${run_idx},DATASET_INDEX=0,num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
            ${NSS_SLURM_DIR}/slurm_srun.sh )
        echo "[${config}] run_index=${run_idx} submitted job ${job_id}"
        prev_job_id="${job_id}"
        echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
      done
    else
      echo "[${config}] Submitting array of ${RUNS} runs (time=${TIME_LIMIT})"
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
          --array=0-$((RUNS-1)) \
          --export=ALL,PHASE=end_to_end,USER_NAME=${USER_NAME},ACCOUNT=llmservice_sdg_research,RUNS=${RUNS},EXP_NAME=${EXP_NAME},DATASET_GROUP=${DATASET_GROUP},CONFIG_NAMES=${config},TIME_LIMIT=${TIME_LIMIT},DATASET_URLS=${DATASET_URLS},num_configs=1,WANDB_PROJECT=${WANDB_PROJECT} \
          ${NSS_SLURM_DIR}/slurm_srun.sh )

      echo "[${config}] submitted job ${job_id}"
      echo "Pausing ${SLEEP_SEC}s before next submit..."; sleep ${SLEEP_SEC}
    fi
  fi
done

echo "Done. Monitor with: squeue -u ${USER_NAME}"




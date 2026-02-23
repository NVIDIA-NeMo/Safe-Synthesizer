#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

# This file will be executed by submit_slurm_jobs.sh and submit_single_dataset.sh

USER_NAME="${USER_NAME:-}"

# Source env_variables.sh - use NSS_SLURM_DIR if available, otherwise construct from USER_NAME
if [ -n "${NSS_SLURM_DIR:-}" ]; then
    source "${NSS_SLURM_DIR}/env_variables.sh"
else
    source /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/nmp/packages/nemo_safe_synthesizer/script/slurm/env_variables.sh
fi

ACCOUNT="${ACCOUNT:-llmservice_sdg_research}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
DATASET_GROUP="${DATASET_GROUP:?DATASET_GROUP env var is required (short|long)}"
RUNS="${RUNS:-5}"
EXP_NAME="${EXP_NAME:-multi_jobs}"
CONFIG_NAMES="${CONFIG_NAMES:-}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SRUN_EXTRA="${SRUN_EXTRA:-}"


echo "Executing slurm_run.sh (DATASET_GROUP=${DATASET_GROUP}, RUNS=${RUNS}, EXP_NAME=${EXP_NAME}, CONFIG_NAMES=${CONFIG_NAMES})"

# Container settings (kept static here, override by editing or via SRUN_EXTRA if needed)
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

srun -A "${ACCOUNT}" \
  --gpus-per-task="${GPUS_PER_TASK}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --time="${TIME_LIMIT}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  ${SRUN_EXTRA} \
  /bin/bash -c "
set -euo pipefail
echo 'Executing slurm_nss_matrix.sh inside container...'
cd ${NMP_DIR}
pwd
export DATASET_GROUP='${DATASET_GROUP}'
export RUNS='${RUNS}'
export EXP_NAME='${EXP_NAME}'
export USER_NAME='${USER_NAME}'
export NSS_SLURM_DIR='${NSS_SLURM_DIR}'
if [ -n '${PHASE:-}' ]; then
  export PHASE='${PHASE}'
fi
if [ -n '${CONFIG_NAMES}' ]; then
  export CONFIG_NAMES='${CONFIG_NAMES}'
fi
bash ${NSS_SLURM_DIR}/slurm_nss_matrix.sh
"

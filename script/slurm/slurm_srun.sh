#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

# This file will be executed by submit_slurm_jobs.sh to queue the slurm job,
# but intentionally has no logic and is just a pass through to
# slurm_nss_matrix.sh.

ACCOUNT="${ACCOUNT:-nemotron_data_dev}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SRUN_EXTRA="${SRUN_EXTRA:-}"
export NEMO_DEPLOYMENT_TYPE="${NEMO_DEPLOYMENT_TYPE:-slurm-nvidia-internal}"

# Pyxis gives variables baked into the container image precedence over the
# Slurm step environment. Preserve the submitted job configuration explicitly
# so a cached image created by another user cannot redirect paths or outputs.
CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS:-\
USER_NAME,LUSTRE_DIR,NSS_SHARED_DIR,NSS_DIR,NSS_SLURM_DIR,CONFIG_DIR,\
BASE_LOG_DIR,ADAPTER_PATH,NSS_ARTIFACTS_PATH,VLLM_CACHE_ROOT,HF_HOME,\
UV_CACHE_DIR,UV_PYTHON_INSTALL_DIR,UV_PYTHON_BIN_DIR,UV_TOOL_DIR,\
UV_CONCURRENT_DOWNLOADS,NSS_PYTHON_VERSION,WANDB_MODE,WANDB_PROJECT,\
NEMO_DEPLOYMENT_TYPE,EXP_NAME,NSS_VERSION,PACKED_DATASETS,PACKED_CONFIGS,NSS_PHASE}"

# No --time flag used in srun since we have a single srun in each allocation so
# the time limit is controleld by the --time flag on the sbatch calls in
# submit_slurm_jobs.sh.
srun -A "${ACCOUNT}" \
  --gpus-per-task="${GPUS_PER_TASK}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-env="${CONTAINER_ENV_VARS}" \
  --export=ALL \
  ${SRUN_EXTRA} \
  /bin/bash -c "
set -euo pipefail
bash ${NSS_SLURM_DIR}/slurm_nss_matrix.sh
"

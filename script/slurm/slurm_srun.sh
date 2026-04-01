#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

# This file will be executed by submit_slurm_jobs.sh or submit_single_dataset.sh
# to queue the slurm job, but intentionally has no logic and is just a pass
# through to slurm_nss_matrix.sh. Env vars are set by the submit scripts and
# propagated via --export=ALL.

ACCOUNT="${ACCOUNT:-llmservice_sdg_research}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SRUN_EXTRA="${SRUN_EXTRA:-}"

# No --time flag used in srun since we have a single srun in each allocation so
# the time limit is controleld by the --time flag on the sbatch calls in
# submit_slurm_jobs.sh.
srun -A "${ACCOUNT}" \
  --gpus-per-task="${GPUS_PER_TASK}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --export=ALL \
  ${SRUN_EXTRA} \
  /bin/bash -c "
set -euo pipefail
bash ${NSS_SLURM_DIR}/slurm_nss_matrix.sh
"

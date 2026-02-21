#!/bin/bash

set -euo pipefail

# This file will be executed by submit_slurm_jobs.sh to queue the slurm job,
# but intentionally has no logic and is just a pass through to
# slurm_nss_matrix.sh.

ACCOUNT="${ACCOUNT:-llmservice_sdg_research}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SRUN_EXTRA="${SRUN_EXTRA:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

srun -A "${ACCOUNT}" \
  --gpus-per-task="${GPUS_PER_TASK}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --time="${TIME_LIMIT}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --export=ALL \
  ${SRUN_EXTRA} \
  /bin/bash -c "
set -euo pipefail
bash ${NSS_SLURM_DIR}/slurm_nss_matrix.sh
"

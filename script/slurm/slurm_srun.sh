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
# Some Pyxis versions treat this as the complete container allowlist, so keep
# the required baseline synchronized with variables consumed by the scripts.
# Cache locations are not listed: slurm_nss_matrix.sh sets them per job.
REQUIRED_CONTAINER_ENV_VARS="\
USER_NAME,LUSTRE_DIR,NSS_SHARED_DIR,NSS_MODELS_DIR,NSS_DIR,NSS_SLURM_DIR,\
CONFIG_DIR,BASE_LOG_DIR,ADAPTER_PATH,NSS_ARTIFACTS_PATH,WANDB_MODE,\
WANDB_PROJECT,NEMO_DEPLOYMENT_TYPE,EXP_NAME,PACKED_DATASETS,PACKED_CONFIGS,\
NSS_PHASE"
CONTAINER_ENV_VARS="${REQUIRED_CONTAINER_ENV_VARS}${CONTAINER_ENV_VARS:+,${CONTAINER_ENV_VARS}}"

# Mounts: Lustre for data and outputs, node-local scratch for caches, and the
# source checkout read-only at its own path. --no-container-mount-home keeps
# Pyxis from mounting the Lustre-backed home at /root, so tools that default
# to ~ write to the container's local filesystem instead.
CONTAINER_MOUNTS="/lustre:/lustre,/raid/scratch:/raid/scratch,${NSS_DIR}:${NSS_DIR}:ro"
# A git worktree keeps its history in the main checkout's .git directory; mount
# it too so the package version can be read from git instead of falling back to 0.0.0.
# Resolved from the .git file and its commondir pointer, without needing git on the host.
if [[ -f "${NSS_DIR}/.git" ]]; then
  worktree_git_dir="$(sed -n 's/^gitdir: //p' "${NSS_DIR}/.git")"
  [[ "${worktree_git_dir}" == /* ]] || worktree_git_dir="${NSS_DIR}/${worktree_git_dir}"
  git_common_dir="$(cd "${worktree_git_dir}" 2>/dev/null && cd "$(cat commondir 2>/dev/null || echo .)" && pwd || true)"
  if [[ -n "${git_common_dir}" && "${git_common_dir}" != "${NSS_DIR}/"* ]]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${git_common_dir}:${git_common_dir}:ro"
  fi
fi

# No --time flag used in srun since we have a single srun in each allocation so
# the time limit is controleld by the --time flag on the sbatch calls in
# submit_slurm_jobs.sh.
# shellcheck disable=SC2086 # SRUN_EXTRA is intentionally word-split into flags
srun -A "${ACCOUNT}" \
  --gpus-per-task="${GPUS_PER_TASK}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --container-image="${NSS_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --no-container-mount-home \
  --container-env="${CONTAINER_ENV_VARS}" \
  --export=ALL \
  ${SRUN_EXTRA} \
  /bin/bash -c "
set -euo pipefail
bash ${NSS_SLURM_DIR}/slurm_nss_matrix.sh
"

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Slurm Node-Local Storage Migration

## Purpose

The MARS cluster team sent a notice asking users not to run Python or any code from Lustre, not to run recursive
`find` from the Lustre root, and to keep databases and caches on node-local disk (`/raid/scratch`) instead of Lustre.
The [MARS Cluster Best Practices](https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/2441645950/Cluster+Best+Practices)
page gives the full policy.

The `script/slurm` workflow breaks all three rules by design: its README requires the interpreter, virtualenv, and
caches to live on Lustre so that containers can see them. This plan replaces that design. It is written to be
self-contained for the session that implements it.

Work happens on a fresh branch, `yunzhang/slurm-node-local-storage`, from `main`. It does not build on
`yunzhang/slurm-bootstrap`.

## Storage policy

| Directory | Holds | Must not hold |
| --- | --- | --- |
| `/home` (10 GB quota; a Lustre filesystem on this cluster) | Scripts, source code, git repos, configuration | Datasets, outputs |
| `/raid/scratch` (node-local) | Builds, caches, databases, lock files, temporary logs | Anything needed after the job |
| `/lustre` | Container images, datasets, checkpoints, large results | Code, builds, databases, caches, millions of small files |

The page shows `HF_HOME` on Lustre as an example of keeping `~/.cache` out of `/home`. The notice bans caches on
Lustre outright, so it takes precedence: all caches go to `/raid/scratch`.

## Current violations

Code on Lustre:

- `.mise/tasks/bootstrap-nss-slurm` installs Python and `.venv` under `$LUSTRE_DIR` and refuses any other location.
- Every array task in `slurm_nss_matrix.sh` imports torch and vLLM from that Lustre venv. In PyPI mode, every task
  also runs `uv venv` and `uv pip install` into one shared Lustre venv at the same time.
- `submit_slurm_jobs.sh` runs `safe-synthesizer --help` from the Lustre venv on the login node.
- `script/slurm/README.md` tells users to clone the repo into `$LUSTRE_DIR`.

Caches and lock files on Lustre:

- `env_variables.sh` and the bootstrap task point `UV_CACHE_DIR`, `HF_HOME`, `VLLM_CACHE_ROOT`, `UV_TOOL_DIR`, and the
  uv Python install at Lustre.
- Pyxis sets `HOME=/root` and mounts the user's Lustre home there, so Triton, W&B, and anything using `~/.cache` write
  to Lustre.
- `cli/wandb_setup.py` passes `dir=workdir.run_dir` to `wandb.init`, so W&B's local run files land in the Lustre run
  directory.
- `submit_slurm_jobs.sh` takes a `flock` on `$LUSTRE_DIR/.locks/nss-repo-venv.lock`.

Other practices the page asks for:

- `nss_top.py` runs two `rglob` scans of `nss_results`, adapters included, every time a row is selected, and polls
  every 30 s instead of 60 s or more.
- When the cached image is missing, `submit_slurm_jobs.sh` falls back to pulling `nvcr.io` on every job.

## Verified cluster facts

Checked on 2026-10-14 from `cs-oci-ord-login-03` with jobs 36579002 (`batch-block7-01961`) and 36579020
(`batch-block5-01672`), using the team's CUDA runtime image:

- `/raid/scratch` exists on GPU nodes: a 19 TB ext4 array with 15 TB free, mode `1777`. It is not wiped between jobs;
  other users' directories from 2026-09-15 were still there.
- Enroot unpacks containers to `/raid/containers/data/user-<uid>`, which is local disk, and the container root
  filesystem is writable.
- Containers run as `root`, mapped back to the submitting user. `$USER` is `root` inside, so scripts must use
  `USER_NAME`.
- Pyxis mounts the user's Lustre home at `/root` and sets `HOME=/root`. `/home/<user>` inside the image is empty.
- `srun` copies the login environment into the job, including `PATH`, `UV_*`, and `HF_HOME` from users' shell files.
- Compute nodes reach PyPI, the PyTorch index, Hugging Face (including its xet download hosts), and `ghcr.io`.
  `cdn-lfs.huggingface.co` has no DNS entry; it is a retired host.
- The current CUDA image has no `python3`, `gcc`, `curl`, or `uv`.
- `training.pretrained_model` accepts a local path, overridable with `--training__pretrained_model`. NSS picks model
  metadata by matching names such as `TinyLlama`, `SmolLM3`, or `Mistral` in the path, so local copies must keep the
  Hugging Face repo name.

## Settled decisions

- All caches go to `/raid/scratch`. The uv and Hugging Face caches are per user and per node, since `/raid/scratch`
  persists; everything else is per job.
- The repo lives at `~/code/Safe-Synthesizer` and is mounted at the same path in the container.
- Dependencies come from one team-shared image, built on the cluster with Pyxis `--container-save` from the existing
  CUDA runtime image. No Docker or registry pull.
- Each job runs `uv sync --frozen` against its branch's `uv.lock`, so branches with changed dependencies work without
  a new image.
- Model weights are a read-only input on Lustre, downloaded once by one owner and copied to `/raid/scratch` by jobs.
- Downloading GitHub release assets during the image build is acceptable. `uv.lock` pins FlashInfer's
  `flashinfer_jit_cache` wheel to a GitHub release URL.
- PyPI mode (`--nss-version`) is removed.
- Git over SSH is fine for developers and agents; avoid the `gh` CLI and API from the cluster.
- No team-shared writable cache in `/raid/scratch`. It would need a team Unix group, `umask 002` in every job, and
  setgid directories on every node, and any teammate's mistake would break other people's jobs on that node only.

## Design

### Image

`script/slurm/build_image.sh` submits one job. It starts from
`$NSS_SHARED_DIR/images/cuda_12_8_1_cudnn_runtime_ubuntu24_04.sqsh`, installs `build-essential`, `git`, `curl`,
`ca-certificates`, and `libgomp1`, and copies in the uv version pinned in `mise.lock` (currently 0.9.30) from the
builder's mise install. It then installs the `.python-version` Python into `/opt/python` and runs
`uv sync --frozen --no-install-project --no-dev --extra engine --extra cu129` into `/opt/venv`, with
`UV_COMPILE_BYTECODE=1` and `UV_LINK_MODE=copy`. Copies of `uv.lock` and `.python-version` go into `/opt/nss-image/`
so the submit script can compare them with the branch.

The job saves the container with `--container-save` to a temporary name in `$NSS_SHARED_DIR/images/`, compressed with
zstd through `ENROOT_SQUASH_OPTIONS`, then renames it to `nss-slurm-py<python>-<first 12 hex of sha256(uv.lock)>.sqsh`.
Existing names are never overwritten, so running jobs are never affected. `NSS_IMAGE` in `env_variables.sh` selects
the default image, and bumping it is a one-line change.

The build mirrors the extras and uv settings of `containers/Dockerfile.cuda`. The two are separate definitions, so
both files should say they must be kept in sync.

Source and config changes never need a new image, and neither do most dependency changes, because jobs sync the
difference. Rebuild when `.python-version` changes, when system packages change, or when a newer `uv.lock` format
needs a newer uv. Rebuilding is also recommended after torch, vLLM, or FlashInfer bumps; otherwise the first job on
each node downloads several GB.

### Job flow

1. At submit time on the login node, check that `$NSS_IMAGE` exists. Warn if the branch's `uv.lock` differs from the
   image's, refuse if `.python-version` differs, then submit a throttled array. There is no venv build, lock file, or
   `apt-get`.
2. `slurm_srun.sh` launches the container from `$NSS_IMAGE` with `--no-container-mount-home` (pending verification)
   and mounts `/lustre`, `/raid/scratch`, and `$NSS_DIR` at the same path.
3. `slurm_nss_matrix.sh` creates `JOB_LOCAL`, sets the environment below, sets a clean `PATH`, and registers a `trap`
   that removes `JOB_LOCAL` on exit.
4. It runs `uv sync --frozen --no-dev --extra engine --extra cu129` from `$NSS_DIR`, which installs any dependency
   differences into `/opt/venv` and the project itself as an editable install from `/home`.
5. If `$NSS_MODELS_DIR/<model id>` exists, the model is copied once per node to `$NODE_CACHE/models/<model id>`
   (copied to a temporary name, then renamed) and passed with `--training__pretrained_model`. Otherwise it downloads
   through the per-node Hugging Face cache.
6. `safe-synthesizer run` writes its outputs to `run_path` on Lustre as today.

### Job environment

| Variable | Value |
| --- | --- |
| `JOB_LOCAL` | `/raid/scratch/$USER_NAME/$SLURM_JOB_ID` |
| `NODE_CACHE` | `/raid/scratch/$USER_NAME/nss-cache` |
| `UV_CACHE_DIR` | `$NODE_CACHE/uv` |
| `UV_PROJECT_ENVIRONMENT` | `/opt/venv` |
| `UV_PYTHON_INSTALL_DIR` | `/opt/python` (overrides any inherited Lustre value) |
| `UV_LINK_MODE` | `copy` |
| `HF_HOME` | `$NODE_CACHE/hf` (the token comes from `HF_TOKEN`) |
| `VLLM_CACHE_ROOT` | `$JOB_LOCAL/vllm` |
| `TRITON_CACHE_DIR` | `$JOB_LOCAL/triton` |
| `TORCHINDUCTOR_CACHE_DIR` | `$JOB_LOCAL/torchinductor` |
| FlashInfer JIT cache | `$JOB_LOCAL/flashinfer` (confirm the variable name for the locked version) |
| `XDG_CACHE_HOME` | `$JOB_LOCAL/xdg-cache` |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR`, `WANDB_DATA_DIR` | Under `$JOB_LOCAL/wandb` |
| `TMPDIR` | `$JOB_LOCAL/tmp` |
| `PYTHONPYCACHEPREFIX` | `$JOB_LOCAL/pycache` |
| `PATH` | `/opt/venv/bin:/usr/local/bin:/usr/bin:/bin` |

The job also unsets `XDG_RUNTIME_DIR`, `UV_PYTHON_BIN_DIR`, `UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`, and `VIRTUAL_ENV`, which
may be inherited from the login shell.

### Models

Models live at `$NSS_MODELS_DIR/<org>/<name>/`, keeping the Hugging Face repo name so NSS recognizes them.
`NSS_MODELS_DIR` defaults to `$NSS_SHARED_DIR/models`. A new `script/slurm/download_models.sh` populates it with
`hf download --local-dir` and then deletes the `.cache/huggingface/` folder that command leaves behind. Jobs only read
the directory; only its owner writes to it.

### W&B

Change `wandb_setup.py` to use `WANDB_DIR` when it is set and fall back to `run_dir` otherwise. `wandb_run_id.txt`
stays in `run_dir`, because the generate phase reads it. In online mode the local W&B files can be discarded at job
end; in offline mode, archive them as one tar file into `run_dir` before cleanup.

### nss_top

Add `STDOUT` and `STDERR` to the existing `squeue --Format` call and open those paths directly instead of scanning
with `rglob`. Raise the default refresh to 60 s.

### Removals

- `.mise/tasks/bootstrap-nss-slurm`, replaced by the image.
- `prebuild_repo_venv` and the `.locks` lock file in `submit_slurm_jobs.sh`.
- Per-job `apt-get` in `slurm_nss_matrix.sh`.
- The `nvcr.io` fallback; a missing image now fails with build instructions.
- PyPI mode, including `--nss-version` and `${LUSTRE_DIR}/.uv`.
- The Lustre `UV_*`, `HF_HOME`, and `VLLM_CACHE_ROOT` settings in `env_variables.sh`.

## File changes

| File | Change |
| --- | --- |
| `script/slurm/env_variables.sh` | Remove Lustre caches; default `NSS_DIR` to `$HOME/code/Safe-Synthesizer`; add `NSS_IMAGE`, `NSS_MODELS_DIR` |
| `script/slurm/submit_slurm_jobs.sh` | Remove prebuild, lock, PyPI mode, `nvcr.io` fallback; add image checks; default throttle |
| `script/slurm/slurm_srun.sh` | Image, mounts, `--no-container-mount-home`, updated container environment allowlist |
| `script/slurm/slurm_nss_matrix.sh` | Job environment, cleanup trap, `uv sync`, model staging; remove `apt-get` and PyPI mode |
| `script/slurm/build_image.sh` | New |
| `script/slurm/download_models.sh` | New |
| `script/slurm/nss_top.py` | Direct log paths, 60 s refresh |
| `src/nemo_safe_synthesizer/cli/wandb_setup.py` | Respect `WANDB_DIR`, with a unit test |
| `.mise/tasks/bootstrap-nss-slurm` | Delete, along with references in docs and tasks |
| `containers/Dockerfile.cuda` | Comment pointing to `build_image.sh` as the second definition |
| `script/slurm/README.md` | Rewrite setup around `~/code`, the image, models, and node-local storage |

## Implementation sequence

1. Run a verification job. Done when `--container-save` writes an image to `images/`, `--no-container-mount-home`
   keeps `~` on local disk, files the container writes to `/raid/scratch` are owned by the user on the host, and
   Pyxis honors `ENROOT_SQUASH_OPTIONS`.
2. Write `build_image.sh` and build the first image. Done when `/opt/venv/bin/python -c "import torch, vllm"`
   succeeds in a job.
3. Update `env_variables.sh`, `slurm_srun.sh`, and `slurm_nss_matrix.sh`. Done when `tinyllama-nodp` succeeds on one
   dataset in `end_to_end` mode.
4. Update `submit_slurm_jobs.sh`. Done when a `two_stage` run succeeds, `--dry-run` works, and a missing image fails
   clearly.
5. Add model staging. Done when jobs use the staged local path and the second job on a node skips the copy.
6. Change `wandb_setup.py`. Done when the unit test passes and an online run leaves only `wandb_run_id.txt` under
   `run_path`.
7. Update `nss_top.py`. Done when logs open without scanning.
8. Apply the removals and rewrite the README. Done when nothing refers to the bootstrap task or PyPI mode.
9. Test a branch with a modified `uv.lock`. Done when the job log shows only the changed packages installing.

## Acceptance checks

- `readlink -f $(which python)` inside a job resolves under `/opt`.
- Non-recursive listings with timestamps of `~`, `~/.cache`, and `$LUSTRE_DIR/.cache` are unchanged by a run.
- `JOB_LOCAL` is gone after a successful job, a failed job, and `scancel`.
- Adapters, synthetic data, and evaluation reports appear under `$ADAPTER_PATH` as before.

## Open items

- Resolved: models and all other shared files live in the project-level shared directory (see implementation
  status). Kendrick's old directory can be retired once he confirms nothing else depends on it.
- Default value for `--max-concurrent-slurm-jobs`; the page recommends throttled arrays.
- A size cap or periodic `uv cache prune` for the per-node caches.
- Whether Pyxis works on `cpu_short` for image builds; otherwise use one GPU on `interactive_singlenode`.

## Implementation status (2026-10-14)

Steps 1 to 8 are implemented on `yunzhang/slurm-node-local-storage`; the branch-dependency test (step 9) and model
staging with a real models directory are not yet run. Differences from the design above:

- Verification (job 36580192 on `cpu-00107`) confirmed `--container-save` honors `ENROOT_SQUASH_OPTIONS`,
  `--no-container-mount-home` keeps `HOME` on local disk, container-written files are owned by the user, and Pyxis
  works on `cpu_short`. CPU nodes have a smaller `/raid/scratch` (253 GB free), enough for builds.
- `build_image.sh` downloads uv inside the container from the URL and checksum in `mise.lock` instead of copying
  the builder's mise install, because `main` requires a newer mise than some users have. The first build (job
  36580425) took 3 minutes and produced `nss-slurm-py3.13-e58529b3b46c.sqsh` (9.6 GB).
- The cluster's `unsquashfs` is 4.4, which has no `-cat`, so `submit_slurm_jobs.sh` extracts `/opt/nss-image` to a
  temporary directory on the login node to compare Python versions and `uv.lock`.
- `JOB_LOCAL` is `/raid/scratch/$USER_NAME/job-<SLURM_JOB_ID>-<pid>`, so manual runs cannot collide with jobs.
- The FlashInfer JIT cache variable is `FLASHINFER_WORKSPACE_BASE`; it defaults to `$HOME`.
- `nss_top.py` gets log paths from `scontrol show job`, which expands `%A_%a` patterns on Slurm 23.02 (the
  `squeue` `STDOUT` field does not), and requires a refresh of at least 60 s. `--log-dir` is removed.
- `env_variables.sh` now respects an exported `WANDB_MODE`.
- A GPU smoke test (`batch-block7-03578`, A100) showed the job-start `uv sync` takes 5 s when `uv.lock` matches the
  image, `python` resolves under `/opt/python`, and CUDA and vLLM work. The first end-to-end run (job 36581840,
  `tinyllama-nodp` on `adult`, W&B disabled) completed in 13 min 45 s, with 49 s from job start to the first NSS
  log line.

- The shared directory moved to the project level:
  `/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer`, group `llmservice`, mode
  `2775`. It holds the base and NSS images, `data/`, `configs/`, `models/`, and a `dataset_registry.yaml` whose
  `base_url` points at the new `data/` (all 54 relative entries resolve). The previous registry still pointed at
  Kendrick's pre-migration `llmservice` portfolio directory. Kendrick's directory was only read, except that our NSS
  image was first published there and then moved out.
- The branch-dependency test (job 36585234, a worktree adding `humanize`) installed exactly the added package and
  the project in about 1 s. It also showed that a worktree checkout reported version `0.0.0` because the main
  `.git` was not mounted; `slurm_srun.sh` now mounts the worktree's common git directory read-only.

- Models are in `shared/safe_synthesizer/models` (SmolLM3-3B 5.6 GB, Mistral-7B-Instruct-v0.3 14 GB without the
  duplicate `consolidated.safetensors`, TinyLlama 1.9 GB). The model-staging test (array 36586079, two tasks on
  different nodes) staged TinyLlama from there, made no Hugging Face downloads, read data from the new `data/`, and
  averaged 799 s per job against 844 s when downloading the model. Skipping the copy on a node that already has the
  model, and two tasks racing to stage on one node, are not yet observed in a run.

### A/B timing against the Lustre setup

Both setups ran `main` at `c8e38eda`: `tinyllama-nodp` on `adult`, `end_to_end`, W&B disabled, two tasks each,
submitted together to `polar4` (old: array 36583803, new: array 36583809). The old setup used a Lustre venv built
with the original bootstrap steps (238 s build), the original scripts, and Lustre caches with TinyLlama already
cached. The new setup downloaded TinyLlama into an empty node cache. Averages per task:

| Segment | Old (Lustre) | New (node-local) | Change |
| --- | --- | --- | --- |
| Job start to first NSS log (container, `apt-get` or `uv sync`) | 65 s | 52 s | -13 s |
| First NSS log to pipeline start (preflight, heavy imports) | 92 s | 20 s | -72 s |
| PII replacement | 113 s | 115 s | same |
| Training | 473 s | 441 s | -31 s (one slow old task: 500 s) |
| Generation and evaluation | 89 s | 84 s | -5 s |
| Rest of pipeline (model loading, vLLM start-up, data preparation) | 187 s | 127 s | -60 s |
| Job total | 1023 s (17 min 3 s) | 844 s (14 min 4 s) | -179 s (-17%) |

With two tasks per side the training difference is within node-to-node noise; the import and start-up savings
(about 145 s) are consistent across all tasks, including an old and a new task that ran on the same node.

## Out of scope

- Personal shell setup. Users' `~/.bash_env` files that export Lustre `UV_*` and `HF_HOME` should move those caches to
  `/tmp` on the login node; job scripts override them regardless.
- Removing old Lustre data. After migration, delete `$LUSTRE_DIR/.cache/{uv,huggingface,vllm}`,
  `$LUSTRE_DIR/.local/share/uv`, `$LUSTRE_DIR/.locks`, and the old checkout's `.venv` gradually rather than in one
  sweep, as the page asks.
- Multi-node jobs; the current workflow runs single-GPU arrays.

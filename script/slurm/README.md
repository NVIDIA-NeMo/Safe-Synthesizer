<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

### NeMo Safe Synthesizer Slurm Jobs

This directory contains scripts to launch Slurm jobs for NeMo Safe Synthesizer experimentation.
The contents of this directly are often specific to internal NVIDIA slurm clusters, but shared here as inspiration for others that might be using slurm to do hyperparameter experiments with NeMo Safe Synthesizer.

Jobs are submitted via `submit_slurm_jobs.sh`, which launches a containerized `srun` (`slurm_srun.sh`) that executes the matrix runner (`slurm_nss_matrix.sh`). All paths and defaults are configured in one place: `env_variables.sh`.

### Where things live

The scripts follow the MARS [cluster storage rules](https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/2441645950/Cluster+Best+Practices): no code, builds, caches, or lock files on Lustre.

| Location | Holds |
| --- | --- |
| `/home` | Your Safe-Synthesizer checkout, mounted read-only at the same path inside jobs |
| Shared NSS image (`.sqsh` on Lustre) | Python, uv, and the locked dependencies in `/opt/venv`; unpacked to node-local disk when a job starts |
| `/raid/scratch/$USER_NAME` (node-local) | uv and Hugging Face caches (kept per node), plus a per-job directory for every other cache, removed when the job ends |
| Lustre | Images, datasets, pre-downloaded models, adapters, results, and logs |

Each job runs `uv sync --frozen` against your checkout's `uv.lock`. If it matches the image, this only installs the project (a few seconds). If your branch changes dependencies, only the differences are installed. A new image is needed only when `.python-version` or system packages change, or after large torch, vLLM, or FlashInfer bumps.

### Files
- `env_variables.sh`: Single source of truth for user, paths, and the image.
- `submit_slurm_jobs.sh`: Submits Slurm array jobs for each config and dataset. Supports two-stage TRAIN→GEN pipeline.
- `slurm_nss_matrix.sh`: Sets up node-local caches, syncs dependencies, picks dataset and config, and launches the python entrypoint inside the container. Honors `NSS_PHASE=train|generate|end_to_end`.
- `slurm_srun.sh`: Wraps `srun` with container image and mounts, mostly just a pass through, primary logic is in `submit_slurm_jobs.sh` and `slurm_nss_matrix.sh`.
- `build_image.sh`: Builds the shared NSS image on the cluster, without Docker.
- `download_models.sh`: Downloads pretrained models once into the shared, read-only models directory.
- `configs/*.yaml`: Major configs we support. Use the config basenames from this directory in commands (for example, `smollm3-nodp`, `smollm3-dp`, etc.). The current set is the cross product of 3 pre-trained models and 2 DP settings (on or off).

Pipeline entrypoints invoked from the image's virtualenv:
- `/opt/venv/bin/safe-synthesizer run --run-path <path>` (full end-to-end pipeline)
- `/opt/venv/bin/safe-synthesizer run train --run-path <path>` (PII replacement + training only)
- `/opt/venv/bin/safe-synthesizer run generate --run-path <path>` (generation + evaluation only)

### Prerequisites

- Slurm Cluster Access: Ensure you have access to the Slurm clusters. You can verify this by running `ssh cs-oci-ord-login-01.nvidia.com` in your terminal (VPN connection required). For an introduction to Slurm, see [these onboarding resources](https://confluence.nvidia.com/display/HWINFCSSUP/Onboarding+to+Clusters).
- An LLM inference endpoint and the API Key: You will need a `NSS_INFERENCE_KEY` to run column classification, if using the default `NSS_INFERENCE_ENDPOINT`. If you do not have one, you can generate it at [build.nvidia.com](https://build.nvidia.com).
- Weights & Biases API Key: W&B logging is enabled by default (`WANDB_MODE=online`). You will need a `WANDB_API_KEY` — request an account [here](https://confluence.nvidia.com/display/AIALGO/Weights+and+Biases+%28WandB%29+Enterprise+Account). Export `WANDB_MODE=disabled` before submitting to skip W&B.
- Enroot Credentials: Follow https://confluence.nvidia.com/display/HWINFCSSUP/Using+Containers#UsingContainers-SettingupEnrootCredentials. You should add the lines for all 3 of `nvcr.io`, `authn.nvidia.com`, and `gitlab-master.nvidia.com`.
- Clone Safe-Synthesizer into your home directory, not Lustre. Git repositories and source code belong in `/home`:

```bash
mkdir -p "$HOME/code" && cd "$HOME/code"
git clone git@github.com:NVIDIA-NeMo/Safe-Synthesizer.git
cd Safe-Synthesizer
```

No Python environment is needed on the login node. Jobs get Python and dependencies from the shared image.

#### Shared image

`env_variables.sh` sets `NSS_IMAGE` to the current shared image. Build a new one only when `submit_slurm_jobs.sh` reports a Python mismatch, or after a large dependency bump:

```bash
bash script/slurm/build_image.sh --wait
```

The build runs as a CPU job (about 10 minutes), names the image after the Python version and a hash of `uv.lock`, publishes it to `$NSS_SHARED_DIR/images/`, and never overwrites an existing image. Point `NSS_IMAGE` in `env_variables.sh` at the new file. Keep `build_image.sh` in sync with `containers/Dockerfile.cuda`.

#### Shared models

Jobs copy pretrained models from `NSS_MODELS_DIR` (default `$NSS_SHARED_DIR/models`) to node-local disk instead of downloading them. Models not found there are downloaded into the per-node Hugging Face cache. To add the models used by `configs/`:

```bash
bash script/slurm/download_models.sh            # all models in configs/*.yaml
bash script/slurm/download_models.sh mistralai/Mistral-7B-Instruct-v0.3
```

#### Nice to have

- Passwordless login See https://confluence.nvidia.com/display/HWINFCSSUP/Setting+Up+Passwordless+SSH+Key+Authentication?src=contextnavpagetreemode
- Env vars in `.bashrc`
  - This is optional, but avoids exporting the submission user in every shell.

```bash
export USER_NAME="<your slurm user name>"
export LUSTRE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/${USER_NAME}"
```


### Before You Run
1) Set your Lustre username, if not already set by `~/.bashrc` (required before submitting):
```bash
export USER_NAME=your_lustre_username
```

2) Create your API token file and restrict permissions. `NSS_INFERENCE_KEY` and `WANDB_API_KEY` are required by default. `HF_TOKEN` is recommended to avoid throttling by HF Hub:
```bash
cat > /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/${USER_NAME}/.api_tokens.sh << 'TOKENS'
export NSS_INFERENCE_KEY="<your_inference_api_key>"
export WANDB_API_KEY="<your_wandb_api_key>"
export HF_TOKEN="<your_hf_token>"
TOKENS
chmod 600 /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/${USER_NAME}/.api_tokens.sh
```

3) Check allocation:
- Review the [Compute Planning spreadsheet](https://docs.google.com/spreadsheets/d/1F6bpK-Z5W5nXKkjKVyEMD9QPw3fJKUcgu0GdXZjZBwQ/edit?gid=757556149#gid=757556149) to confirm available resources and planned usage.
- Monitor current GPU usage in the [AI Hub Dashboard](https://aihub.nvidia.com/) (~3hr delay):
    - Navigate to Observability > GPU Occupancy Trends.
    - Select the cluster: `cs-oci-ord` (primary cluster for NSS experiments).
    - Filter by account using the regex: `nemotron`.
    - Set the interval to 1 hour for a detailed view.
- Use `sshare -U $USER_NAME -l` to check your instantaneous [Fair Share](https://confluence.nvidia.com/display/HWINFCSSUP/Fairshare+Deep+dive) (FS) on a cluster


### Configure
Edit `env_variables.sh` to match your environment. Key items:
- `CONFIGS=(...)`: base names of YAML configs to run (without `.yaml`), or provide via `--configs` argument to `submit_slurm_jobs.sh`.
- `CONFIG_DIR`: directory where config files live.
- `BASE_LOG_DIR`: where Slurm logs will be written.
- `NSS_DIR`: path to this repository; defaults to the checkout containing `env_variables.sh`.
- `ADAPTER_PATH`: base path for workdirs (each run creates a subdirectory with adapter, logs, and outputs).
- `NSS_IMAGE`: shared NSS image that jobs run in; built by `build_image.sh`.
- `NSS_MODELS_DIR`: read-only directory of pre-downloaded models (`<org>/<name>`).
- Cache locations are not configured here. `slurm_nss_matrix.sh` points every cache at node-local `/raid/scratch` for each job.
- `NSS_SHARED_DIR`: location of shared files such as benchmark data and container images, see section below for details.

NSS CLI Environment Variables (used by `safe-synthesizer` CLI via pydantic-settings):
- `NSS_ARTIFACTS_PATH`: Base directory for artifacts (aliased from `ADAPTER_PATH`).
- `NSS_PHASE`: Current phase (train, generate, end_to_end).
- `NSS_CONFIG`: Path to YAML config file.
- `NSS_LOG_FORMAT`: Log format ("json" or "plain").
- `NSS_LOG_FILE`: Path to log file.

Note: Associative arrays/arrays aren't exported to child processes, so only `submit_slurm_jobs.sh` uses them directly.
When needed, arrays are converted to a comma delimited value in an environment variable to pass through to `slurm_nss_matrix.sh`.
This is used for `PACKED_DATASETS` and `PACKED_CONFIGS` which contain the information for all jobs within the array.
In `slurm_nss_matrix.sh`, each job extracts the dataset and config that it should run based on the `SLURM_ARRAY_TASK_ID` environment variable.

### Submit jobs

Run the submit script (flags are order-independent) from this directory:

```bash
bash submit_slurm_jobs.sh [--configs c1,c2] [--dataset-urls name1,url1,path1] [--dataset-group short|long] [--runs N] [--exp-name NAME] [--pipeline-mode two_stage|end_to_end] [--partition P] [--wandb-project PROJECT] [--max-concurrent-slurm-jobs N] [--time-limit TIME] [--train-time-limit TIME] [--generate-time-limit TIME] [--dry-run]

# Example: end_to_end with 2 hour time limit across "short" datasets
bash submit_slurm_jobs.sh --exp-name short_end_to_end --dataset-group short --runs 1 --partition polar4 --pipeline-mode end_to_end --time-limit 2:00:00

# Example: two-stage (TRAIN→GEN) across "short" datasets with 1 hour train time limit and 30 minute generate time limit
bash submit_slurm_jobs.sh --exp-name short_two_stage --dataset-group short --runs 1 --partition polar4 --pipeline-mode two_stage  --train-time-limit 1:00:00 --generate-time-limit 0:30:00

# Example: Adult data (defined in NVIDIA internal dataset_registry.yaml), two configs, 5 runs each on polar4, use different wandb project from the exp name
bash submit_slurm_jobs.sh \
  --dataset-urls adult \
  --configs smollm3-nodp,smollm3-dp \
  --runs 5 \
  --partition polar4 \
  --exp-name regex_adult \
  --pipeline-mode two_stage \
  --wandb-project other_adult

# Example: arbitrary path/url (not a named dataset from the dataset_registry.yaml), 1 config, 10 runs, with max 3 jobs running at a time
bash submit_slurm_jobs.sh \
  --dataset-urls "https://raw.githubusercontent.com/gretelai/gretel-blueprints/refs/heads/main/sample_data/financial_transactions.csv" \
  --configs tinyllama-nodp \
  --runs 10 \
  --partition polar,polar3,polar4 \
  --exp-name financial_repeats \
  --pipeline-mode end_to_end \
  --max-concurrent-slurm-jobs 3
```

- CONFIGS source: By default, configs come from `CONFIGS=(...)` in `env_variables.sh`. Override with `--configs c1,c2` (base names without `.yaml`).
- `--runs`: Number of runs per dataset-config pair.
- `--partition`: Slurm partition(s) to use. See partition info in your cluster docs.
- `--exp-name`: Experiment namespace for logs/outputs.
- `--dataset-group`: `short` or `long` (selects built-in dataset sets).
  Mutually exclusive with `--dataset-urls`.
- `--dataset-urls`: comma separated value of named datasets from registry, file path, or url
  Mutually exclusive with `--dataset-group`.
- `--pipeline-mode`: `two_stage` (TRAIN→GEN with dependency) or `end_to_end` (single job).
- `--wandb-project`: Name of the Weights & Biases project to track experiments.
  Defaults to `--exp-name` if not specified.


### How many jobs will run concurrently?

In general, concurrent jobs will depend on the cluster GPU availability and the Fair Share for the PPP.

- In `two_stage` mode, the submitter launches one TRAIN array and one GENERATE array. GENERATE tasks are linked to corresponding TRAIN tasks via `aftercorr`. Effective max concurrency is cluster/partition limited, but GEN tasks won’t start until their matching TRAIN tasks succeed.
- In `end_to_end` mode, a single array is submitted of size `# datasets * runs * # configs`.

The `--max-concurrent-slurm-jobs N` param can be used to further restrict concurrent jobs.
This only restricts within an array, so with end_to_end mode, this will restrict to precisely N simultaneously running jobs.
In two_stage mode, up to 2*N jobs might run, N each from TRAIN arrays and GENERATE arrays.
Using `--max-concurrent-slurm-jobs` is recommended for large experiments to reduce bursting and be friendlier to other users.
Consider using a max of 2-3x the current allocation for nemotron_data_dev PPP in the cluster to avoid bursting and rapidly dropping our Fair Share for everyone.

### Logs and outputs
- Slurm logs: `${BASE_LOG_DIR}/${EXP_NAME}/slurm_%A_%a.{out,err}`
- You can tail logs while jobs run:
```bash
tail -f ${BASE_LOG_DIR}/${EXP_NAME}/slurm_*.out
```
- W&B logging: `WANDB_MODE` is set to `online` by default to additionally log experiment configs and metrics to W&B. Make sure to export your `WANDB_API_KEY` (request an account [here](https://confluence.nvidia.com/display/AIALGO/Weights+and+Biases+%28WandB%29+Enterprise+Account)) in `${LUSTRE_DIR}/.api_tokens.sh`. There is an optional flag `--wandb-project` to specify a W&B project name if you don't want to use the experiment name.

  - When running in `two_stage` mode, be mindful not to submit multiple bash commands that run simutaneously because we aren't able to guarantee unique adapter path for each single run. As a result, two runs might be logged as one on W&B.

### Monitoring and cancellation
```bash
squeue -u ${USER_NAME}
scancel <jobid>
```

#### nss_top — interactive TUI monitor

`nss_top.py` is a `k9s`-style terminal dashboard for watching your SLURM jobs and tailing their logs in real time. Run it from the login node:

```bash
# Keep uv's cache off Lustre on the login node
export UV_CACHE_DIR="/tmp/${USER}/uv-cache"

# Username is inferred from $USER_NAME or $USER
uv run script/slurm/nss_top.py

# Override username or refresh interval (60 seconds minimum, per cluster guidance)
uv run script/slurm/nss_top.py --user mkornfield --refresh 120
```

Key bindings:

| Key | Action |
|-----|--------|
| `↑` / `↓` | Select job |
| `l` | Show stdout log |
| `e` | Show stderr log |
| `r` | Manual refresh |
| `q` | Quit |

Log paths come from `scontrol show job`, so `nss_top` never searches the filesystem for log files.

### Collect results

W&B is enabled by default with `WANDB_MODE=online` in `env_variables.sh`. Make sure to add your W&B token to `.api_tokens.sh`. Set `WANDB_MODE=disabled` otherwise.

### Troubleshooting

- "USER_NAME is not set": run `export USER_NAME=...` and retry.
- Missing token file/key: create `${LUSTRE_DIR}/.api_tokens.sh` with `NSS_INFERENCE_KEY` and `chmod 600`.
- Missing config files: verify `CONFIGS` in `env_variables.sh` and files in `CONFIG_DIR`.
- Permission errors: confirm your `/lustre/.../${USER_NAME}` paths and file perms.
- `NSS image not found` or a Python mismatch at submit: build a new image with `build_image.sh` and update `NSS_IMAGE`.
- Slow job start after changing `uv.lock`: the first job on each node installs the changed packages; later jobs reuse the node's uv cache. For large changes, build a new image.

#### cpu bind errors

Observed output in *.err file:
```
srun: error: CPU binding outside of job step allocation, allocated CPUs are: 0x0000000F0000000F0000000F0000000F.
srun: error: Task launch for StepId=7993406.0 failed on node pool0-00509: Unable to satisfy cpu bind request
srun: error: Application launch failed: Unable to satisfy cpu bind request
srun: Job step aborted
```

Cause: submitting a job from within a slurm job, i.e., an interactive bash session.
Solution: Only submit slurm jobs from the login or vscode nodes. (May be ways to change some environment variables to resolve, but better to just submit from login node.)

### NSS Shared Directory

To reduce duplicated files and make getting started a bit easier, we have a shared directory for common files that do not change across experiments and the people running them.
It lives at the project level, `/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer`, so it does not depend on any one person's account.
It is owned by group `llmservice` with setgid directories (mode `2775`), so every team member can add and remove files, and new files inherit the group.
Keep new files group-writable (`umask 002`); `build_image.sh` and `download_models.sh` already do this.
It currently exists on `cs-oci-ord`; duplicate it to other clusters as needed (see below).

The `env_variables.sh` script sets the `NSS_SHARED_DIR` variable to provide access to this location. The structure is:

- $NSS_SHARED_DIR
  - dataset_registry.yaml
  - images
    - cuda_12_8_1_cudnn_runtime_ubuntu24_04.sqsh - base image for `build_image.sh`
    - nss-slurm-py<python>-<lockhash>.sqsh - NSS images that jobs run in
  - models
    - <org>/<name> - pre-downloaded models from `download_models.sh`
  - data
    - cleaned
      - <benchmark and other useful datasets for testing>

These resources are used by the slurm scripts in the following ways:
- Jobs run in an NSS image from `$NSS_SHARED_DIR/images`; images are never pulled from a registry per job.
- `dataset_registry.yaml` is passed to safe synthesizer via `--dataset-registry` to enable referencing datasets by name, and for the `submit_slurm_jobs.sh` script to work properly.
  Add additional named datasets and any config overrides to this YAML file as needed.



#### Duplicate shared directory to a new cluster

Create the directory with the team group and setgid first, then copy from `cs-oci-ord` on a file copier node (not a login node).
Copying without preserving owner and group lets files inherit `llmservice` from the setgid directories.

```
umask 002
mkdir -p /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer
chgrp llmservice /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer
chmod 2775 /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer
rsync -rltP --chmod=Dug+rwx,Do+rx,Fug+rw,Fo+r cs-oci-ord-001-dc-02.cs-oci-ord-001.hpc.nvidia.com:/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer/ /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/shared/safe_synthesizer/
```

`dataset_registry.yaml` sets `base_url` to the absolute `data/` path, so update it if the path differs on the new cluster.

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

### NeMo Safe Synthesizer Slurm Jobs

This directory contains scripts to launch Slurm jobs for NeMo Safe Synthesizer experimentation.
The contents of this directly are often specific to internal NVIDIA slurm clusters, but shared here as inspiration for others that might be using slurm to do hyperparameter experiments with NeMo Safe Synthesizer.

Jobs are submitted via `submit_slurm_jobs.sh`, which launches a containerized `srun` (`slurm_srun.sh`) that executes the matrix runner (`slurm_nss_matrix.sh`). All paths and defaults are configured in one place: `env_variables.sh`.

### Files
- `env_variables.sh`: Single source of truth for user, paths.
- `submit_slurm_jobs.sh`: Submits Slurm array jobs for each config and dataset. Supports two-stage TRAIN→GEN pipeline.
- `slurm_nss_matrix.sh`: Picks dataset and config and launches the python entrypoint inside the container. Honors `NSS_PHASE=train|generate|end_to_end`.
- `slurm_srun.sh`: Wraps `srun` with container image and mounts, mostly just a pass through, primary logic is in `submit_slurm_jobs.sh` and `slurm_nss_matrix.sh`.
- `.mise/tasks/bootstrap-nss-slurm`: Installs a container-visible Python and project virtualenv under the current user's Lustre directory.
- `configs/*.yaml`: Major configs we support. Use the config basenames from this directory in commands (for example, `smollm3-nodp`, `smollm3-dp`, etc.). The current set is the cross product of 3 pre-trained models and 2 DP settings (on or off).

Pipeline entrypoints invoked from the prebuilt project virtualenv:
- `.venv/bin/safe-synthesizer run --run-path <path>` (full end-to-end pipeline)
- `.venv/bin/safe-synthesizer run train --run-path <path>` (PII replacement + training only)
- `.venv/bin/safe-synthesizer run generate --run-path <path>` (generation + evaluation only)

### Prerequisites

- Slurm Cluster Access: Ensure you have access to the Slurm clusters. You can verify this by running `ssh cs-oci-ord-login-01.nvidia.com` in your terminal (VPN connection required). For an introduction to Slurm, see [these onboarding resources](https://confluence.nvidia.com/display/HWINFCSSUP/Onboarding+to+Clusters).
- An LLM inference endpoint and the API Key: You will need a `NSS_INFERENCE_KEY` to run column classification, if using the default `NSS_INFERENCE_ENDPOINT`. If you do not have one, you can generate it at [build.nvidia.com](https://build.nvidia.com).
- Weights & Biases API Key: W&B logging is enabled by default (`WANDB_MODE=online`). You will need a `WANDB_API_KEY` — request an account [here](https://confluence.nvidia.com/display/AIALGO/Weights+and+Biases+%28WandB%29+Enterprise+Account). Set `WANDB_MODE=disabled` in `env_variables.sh` to skip W&B.
- Enroot Credentials: Follow https://confluence.nvidia.com/display/HWINFCSSUP/Using+Containers#UsingContainers-SettingupEnrootCredentials. You should add the lines for all 3 of `nvcr.io`, `authn.nvidia.com`, and `gitlab-master.nvidia.com`.
- Clone Safe-Synthesizer

The instructions below assume that Safe-Synthesizer is cloned directly under
`LUSTRE_DIR` and that commands after cloning are run from the repository root.

```bash
export USER_NAME="$USER" # Or hardcode username in slurm
export LUSTRE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/${USER_NAME}"
cd "${LUSTRE_DIR}"
git clone git@github.com:NVIDIA-NeMo/Safe-Synthesizer.git
cd Safe-Synthesizer
```

#### Bootstrap the Slurm Python environment

Do not use the general development bootstrap for Slurm. Slurm containers mount
`/lustre`, but may not have access to the login node's `/home` directory. The
Python interpreter, project virtualenv, package caches, and model caches must
therefore resolve under Lustre.

From the repository root, install the repository-pinned mise tools and run the
Slurm-specific bootstrap:

```bash
make install-mise
export PATH="${HOME}/.local/bin:${PATH}"
export MISE_IGNORED_CONFIG_PATHS="${HOME}/.config/mise/config.toml"
mise install --locked
MISE_LOCKED=1 mise run bootstrap-nss-slurm cu129
```

Existing Slurm checkouts should run this bootstrap once after pulling the
change. Scheduled launch environments, including GitLab jobs, must also add
`${HOME}/.local/bin` to `PATH` before invoking `submit_slurm_jobs.sh`. The task
recreates the repo `.venv` if it points outside Lustre; old Python or uv
installs under `/home` do not need to be removed.

The task derives the Slurm username from `id -un`, uses mise's pinned `uv`,
installs the pinned Python under the user's Lustre directory, and creates
`.venv` with that exact interpreter. If an existing `.venv` points outside
Lustre, the task recreates it. Existing Python and uv installations under
`/home` do not need to be removed.

Override the inferred user or Lustre directory only when the cluster account
layout requires it:

```bash
NSS_SLURM_USER="your_user" MISE_LOCKED=1 mise run bootstrap-nss-slurm cu129
NSS_LUSTRE_DIR="/custom/lustre/path" MISE_LOCKED=1 mise run bootstrap-nss-slurm cu129
```

Verify the postcondition before submitting jobs:

```bash
readlink -f .venv/bin/python
.venv/bin/safe-synthesizer --help >/dev/null
```

The Python path must start with `/lustre/fsw/` or its canonical `/lustre/fs11/`
equivalent. `cu129` and `cuda` select the same CUDA 12.9 dependency profile.

Repo mode, the default when `--nss-version` is omitted, does not require a
separate uv installation under Lustre. PyPI mode uses uv inside the container
and still requires `${LUSTRE_DIR}/.uv/bin/env`.

For PyPI mode only, install the same uv version that mise has pinned:

```bash
export USER_NAME="${USER_NAME:-$(id -un)}"
export LUSTRE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/${USER_NAME}"
uv_version="$(mise exec -- uv --version | awk '{print $2}')"
curl -LsSf "https://astral.sh/uv/${uv_version}/install.sh" \
  | env UV_INSTALL_DIR="${LUSTRE_DIR}/.uv/bin" sh
```

#### Nice to have

- Passwordless login See https://confluence.nvidia.com/display/HWINFCSSUP/Setting+Up+Passwordless+SSH+Key+Authentication?src=contextnavpagetreemode
- Env vars in `.bashrc`
  - This is optional for bootstrap, but avoids exporting the submission user in every shell.

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
- `NSS_DIR`: path to this repository.
- `ADAPTER_PATH`: base path for workdirs (each run creates a subdirectory with adapter, logs, and outputs).
- `VLLM_CACHE_ROOT`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PYTHON_BIN_DIR`, `UV_TOOL_DIR`, `HF_HOME`: cache locations to avoid stressing login nodes.
- `NSS_PYTHON_VERSION`: Python version used for Slurm virtualenvs. Defaults to the repo's pinned version from `.python-version` (so it tracks Python bumps automatically) and falls back to `3.13`. For PyPI mode it is also included in the cached venv name to avoid reusing venvs created with older Python versions. Export it before sourcing `env_variables.sh` to override.
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

> **Run one submit at a time.** `submit_slurm_jobs.sh` builds the shared
> `${NSS_DIR}/.venv` on the login node before submitting. Running multiple submits
> concurrently (especially from different login nodes, where the build lock is not
> reliable across hosts) can corrupt that venv and lets branch switches race. Let
> one submit finish before starting another.

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
# Simplest — username and log dir are inferred from $USER_NAME / $BASE_LOG_DIR / $LUSTRE_DIR
uv run script/slurm/nss_top.py

# Explicit log dir (searches recursively, so the top-level nss_results dir is fine)
uv run script/slurm/nss_top.py --log-dir ${BASE_LOG_DIR}

# Override username or refresh interval
uv run script/slurm/nss_top.py --user mkornfield --refresh 15
```

Key bindings:

| Key | Action |
|-----|--------|
| `↑` / `↓` | Select job |
| `l` | Show stdout log |
| `e` | Show stderr log |
| `r` | Manual refresh |
| `q` | Quit |

Log directory resolution order (first match wins):
1. `--log-dir` flag
2. `$BASE_LOG_DIR` environment variable
3. `$LUSTRE_DIR/nss_results` (constructed from `$LUSTRE_DIR`)
4. `/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/<user>/nss_results` (default)

### Collect results

W&B is enabled by default with `WANDB_MODE=online` in `env_variables.sh`. Make sure to add your W&B token to `.api_tokens.sh`. Set `WANDB_MODE=disabled` otherwise.

### Troubleshooting

- "USER_NAME is not set": run `export USER_NAME=...` and retry.
- Missing token file/key: create `${LUSTRE_DIR}/.api_tokens.sh` with `NSS_INFERENCE_KEY` and `chmod 600`.
- Missing config files: verify `CONFIGS` in `env_variables.sh` and files in `CONFIG_DIR`.
- Permission errors: confirm your `/lustre/.../${USER_NAME}` paths and file perms.
- Virtualenv Python under `/home`: run `MISE_LOCKED=1 mise run bootstrap-nss-slurm cu129`; the task recreates `.venv` with a Lustre-managed interpreter.

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
At this time, the best recommendation is to place this in someone's user directory, so Kendrick created `/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/kendrickb/shared_safe_synthesizer` on the `cw-pdx-cs` and `cs-oci-ord` clusters.
We will want to duplicate this to other clusters that we use.

The `env_variables.sh` script sets the `NSS_SHARED_DIR` variable to provide access to this location. The structure is:

- $NSS_SHARED_DIR
  - dataset_registry.yaml
  - images
    - cuda_12_8_1_cudnn_runtime_ubuntu24_04.sqsh - container image used by current scripts
  - data
    - cleaned
      - <benchmark and other useful datasets for testing>

These resources are used by the slurm scripts in the following ways:
- Cuda image used for slurm jobs is pulled from `$NSS_SHARED_DIR/images` if possible, this improves job startup time and eliminates errors while pulling the image over the network.
- `dataset_registry.yaml` is passed to safe synthesizer via `--dataset-registry` to enable referencing datasets by name, and for the `submit_slurm_jobs.sh` script to work properly.
  Add additional named datasets and any config overrides to this YAML file as needed.



#### Duplicate shared directory to a new cluster

From a file copier node on the new cluster, run the following to copy Kendrick's shared directory from `cs-oci-ord`. Took ~30 minutes when run in Jan 2026 to copy 16 GB.

```
rsync -avzP cs-oci-ord-001-dc-02.cs-oci-ord-001.hpc.nvidia.com:/lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/kendrickb/shared_safe_synthesizer/ /lustre/fsw/portfolios/nemotron/projects/nemotron_data_dev/users/kendrickb/shared_safe_synthesizer/
```

Also good to check on ownership and permissions after copying to ensure 775 permissions (for directories) or 664 (for files).

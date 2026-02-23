<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

### NeMo Safe Synthesizer Slurm Jobs

This directory contains scripts to launch matrix Slurm jobs for NeMo Safe Synthesizer. Jobs are submitted via `submit_slurm_jobs.sh`, which launches a containerized `srun` (`slurm_srun.sh`) that executes the matrix runner (`slurm_nss_matrix.sh`). All paths and defaults are configured in one place: `env_variables.sh`.

### Files
- `env_variables.sh`: single source of truth for user, paths, configs, and time limits.
- `submit_slurm_jobs.sh`: submits Slurm array jobs for each config and dataset group. Supports two-stage TRAIN→GEN pipeline.
- `submit_single_dataset.sh`: submits jobs for one dataset/path. Supports two-stage TRAIN→GEN pipeline.
- `slurm_srun.sh`: wraps `srun` with container image and mounts.
- `slurm_nss_matrix.sh`: picks dataset/index/run and launches the python entrypoint inside the container. Honors `PHASE=train|generate`.

Pipeline entrypoints (invoked by Slurm scripts) via uv:
- `uv run safe-synthesizer run --run-path <path>` (full end-to-end pipeline)
- `uv run safe-synthesizer run train --run-path <path>` (PII replacement + training only)
- `uv run safe-synthesizer run generate --run-path <path>` (generation + evaluation only)

### Prerequisites

- Slurm Cluster Access: Ensure you have access to the Slurm clusters. You can verify this by running `ssh cs-oci-ord-login-01.nvidia.com` in your terminal (VPN connection required). For an introduction to Slurm, see [these onboarding resources](https://confluence.nvidia.com/display/HWINFCSSUP/Onboarding+to+Clusters).
- NIM API Key: You will need a `NIM_API_KEY` to run column classification. If you do not have one, you can generate it at [build.nvidia.com](https://build.nvidia.com) using your `nvidian` organization account.
- Enroot Credentials Follow https://confluence.nvidia.com/display/HWINFCSSUP/Using+Containers#UsingContainers-SettingupEnrootCredentials. You should add the lines for all 3 of `nvcr.io`, `authn.nvidia.com`, and `gitlab-master.nvidia.com`.
- uv and python install in the slurm cluster
  - This is a strongly recommended setup, but is not be the only way to get things working.
  - The key issues about working in slurm we need to address
    - /home/$USER is quite small (10 GB) and not recommended for accessing data, easily filled up by uv cache
    - Slurm jobs may run in containers with different $HOME (and different users/uids)
  - Thus we put uv and python in your user directory in /lustre and not in /home/$USER
```bash
export USER_NAME="$USER" # Or hardcode username in slurm
export LUSTRE_DIR="/lustre/fsw/portfolios/llmservice/users/${USER_NAME}"
# Install `uv` to your lustre directory
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$LUSTRE_DIR/.uv/bin" sh
# Set environment variables so `uv` uses $LUSTRE_DIR subdirectories for storage
export UV_CACHE_DIR="${LUSTRE_DIR}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${LUSTRE_DIR}/.local/share/uv/python"
export UV_PYTHON_BIN_DIR="${LUSTRE_DIR}/.local/bin"
export UV_TOOL_DIR="${LUSTRE_DIR}/.local/share/uv/tools"
# Install python 3.11 (as required by NSS) in a location `uv` is aware of
uv python install 3.11
```

#### Nice to have

- Passwordless login See https://confluence.nvidia.com/display/HWINFCSSUP/Setting+Up+Passwordless+SSH+Key+Authentication?src=contextnavpagetreemode
- Env vars in .bashrc
  - Add `export VARIABLE=VALUE` to the end of `~/.bashrc` for commonly used environment variables, like `USER_NAME` and `LUSTRE_DIR`.
  - Recommended snippet to have in `~/.bashrc` so uv and python work on login node and slurm jobs:
```bash
export USER_NAME=<your slurm user name>
export LUSTRE_DIR="/lustre/fsw/portfolios/llmservice/users/${USER_NAME}"

# (May be added automatically by uv)
. "/lustre/fsw/portfolios/llmservice/users/kendrickb/.uv/bin/env"

export UV_CACHE_DIR="${LUSTRE_DIR}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${LUSTRE_DIR}/.local/share/uv/python"
export UV_PYTHON_BIN_DIR="${LUSTRE_DIR}/.local/bin"
export UV_TOOL_DIR="${LUSTRE_DIR}/.local/share/uv/tools"
export HF_HOME="${LUSTRE_DIR}/.cache/huggingface"
```


### Before You Run
1) Set your Lustre username, if not already set by `~/.bashrc` (required before submitting):
```bash
export USER_NAME=your_lustre_username
```

2) Create your API token file with `NIM_API_KEY` and restrict permissions, recommended to inclue `HF_TOKEN` to avoid throttling by HF Hub and, if you're using W&B, `WANDB_API_KEY`:
```bash
echo 'export NIM_API_KEY="<your_api_key>"' > /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/.api_tokens.sh
chmod 600 /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/.api_tokens.sh
```

3) Check allocation:
- Review the [Compute Planning spreadsheet](https://docs.google.com/spreadsheets/d/1F6bpK-Z5W5nXKkjKVyEMD9QPw3fJKUcgu0GdXZjZBwQ/edit?gid=757556149#gid=757556149) to confirm available resources and planned usage.
- Monitor current GPU usage in the [AI Hub Dashboard](https://aihub.nvidia.com/) (~3hr delay):
    - Navigate to Observability > GPU Occupancy Trends.
    - Select the cluster: `cs-oci-ord` (primary cluster for NSS experiments), `cw-dfw-cs-001`
    - Filter by account using the regex: `sdg`.
    - Set the interval to 1 hour for a detailed view.
- Use `sshare -U $USER_NAME -l` to check your instanteous [Fair Share[(https://confluence.nvidia.com/display/HWINFCSSUP/Fairshare+Deep+dive)] (FS) on a cluster


### Configure
Edit `env_variables.sh` to match your environment. Key items:
- `CONFIGS=(...)`: base names of YAML configs to run (without `.yaml`).
- `CONFIG_DIR`: directory where config files live.
- `BASE_LOG_DIR`: where Slurm logs will be written.
- `NMP_DIR`: path to this repository.
- `ADAPTER_PATH`: base path for workdirs (each run creates a subdirectory with adapter, logs, and outputs).
- `VLLM_CACHE_ROOT`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PYTHON_BIN_DIR`, `UV_TOOL_DIR`, `HF_HOME`: cache locations to avoid stressing login nodes.
- `NSS_SHARED_DIR`: location of shared files such as benchmark data and container images, see section below for details.
- Time limits: `CONFIG_TIME_LIMITS_SHORT` and `CONFIG_TIME_LIMITS_LONG` associative maps. Keys are matched by pattern (`unsloth`, `dp`), falling back to `max`.

NSS CLI Environment Variables (used by `safe-synthesizer` CLI via pydantic-settings):
- `NSS_ARTIFACTS_PATH`: Base directory for artifacts (aliased from `ADAPTER_PATH`).
- `NSS_PHASE`: Current phase (train, generate, end_to_end).
- `NSS_CONFIG`: Path to YAML config file.
- `NSS_LOG_FORMAT`: Log format ("json" or "plain").
- `NSS_LOG_FILE`: Path to log file.

Note: Associative arrays/arrays aren't exported to child processes, so only `submit_slurm_jobs.sh` uses them directly.

### Submit jobs (matrix across dataset groups)
Run the matrix submitter (flags are order-independent) from this directory:
```bash
bash submit_slurm_jobs.sh [--configs c1,c2] [--runs N] [--partition P] [--exp-name NAME] [--dataset-group short|long] [--sleep-sec S] [--pipeline-mode two_stage|end_to_end] [--submit-mode array|sequential] [--wandb-project PROJECT]

# Example: two-stage (TRAIN→GEN) across "short" datasets (array mode; default)
bash submit_slurm_jobs.sh --exp-name matrix_exp --dataset-group short --runs 1 --partition polar4 --pipeline-mode two_stage

# Example: two-stage (TRAIN→GEN) sequential per dataset/run (GEN after TRAIN)
bash submit_slurm_jobs.sh --exp-name matrix_seq --dataset-group short --runs 1 --partition polar4 --pipeline-mode two_stage --submit-mode sequential

# Example: end-to-end (single job per run) sequential per dataset/run
bash submit_slurm_jobs.sh --exp-name matrix_e2e_seq --dataset-group short --runs 1 --partition polar4 --pipeline-mode end_to_end --submit-mode sequential
```

- CONFIGS source: By default, configs come from `CONFIGS=(...)` in `env_variables.sh`. Override with `--configs c1,c2` (base names without `.yaml`).
- RUNS: Number of runs per dataset-config pair.
- PARTITION: Slurm partition to use. See partition info in your cluster docs.
- `EXP_NAME`: Experiment namespace for logs/outputs.
- `DATASET_GROUP`: `short` or `long` (selects built-in dataset sets and time limits).
- `SLEEP_SEC`: Pause between submissions to reduce image import contention.
- `PIPELINE_MODE`: `two_stage` (TRAIN→GEN with dependency) or `end_to_end` (single job).
- `SUBMIT_MODE`: `array` (submit arrays) or `sequential` (submit jobs with dependencies per dataset/run).
- `WANDB_PROJECT`: Name of the Weights & Biases project to track experiments. Defaults to the experiment name if not specified.

How many jobs will run concurrently?

For the built-in `short` group there are currently 17 datasets (see `slurm_nss_matrix.sh`).
- In `two_stage` mode with arrays, the submitter launches one TRAIN array and one GEN array. GEN tasks are linked to corresponding TRAIN tasks via `aftercorr`. Effective max concurrency is cluster/partition limited, but GEN tasks won’t start until their matching TRAIN tasks succeed.
- In `end_to_end` mode, a single array is submitted of size `num_datasets * RUNS * NUM_CONFIGS`.

How long will my jobs take?
With `num_input_records_to_sample=25000`
- For the baseline config, the longest job typically finishes within 80 minutes. Total wall time estimate: `60 * RUNS` minutes.
- For the `dp` config, the longest job typically finishes within 120 minutes. Total wall time estimate: `120 * RUNS` minutes.


### Logs and outputs
- Slurm logs: `${BASE_LOG_DIR}/${EXP_NAME}/short|long/<config>/slurm_%A_%a.{out,err}`
- You can tail logs while jobs run:
```bash
tail -f ${BASE_LOG_DIR}/${EXP_NAME}/short/<config>/slurm_*.out
```
- W&B logging: set the `WANDB_MODE` to `online` to additionally log experiment configs and metrics to W&B. Make sure to export your `WANDB_API_KEY` (request an account [here](https://confluence.nvidia.com/display/AIALGO/Weights+and+Biases+%28WandB%29+Enterprise+Account)) in `${LUSTRE_DIR}/.api_tokens.sh`. There is an optional flag `--wandb-project` to specify a W&B project name if you don't want to use the experiment name.

  - When running in `two_stage` mode, be mindful not to submit multiple bash commands that run simutaneously because we aren't able to guarantee unique adapter path for each single run. As a result, two runs might be logged as one on W&B.

### One-off single dataset runs
For quick testing of a specific CSV with selected configs and N runs, run from this directory using `submit_single_dataset.sh`.

Usage:
```bash
bash submit_single_dataset.sh --dataset-urls PATH_OR_URL [--configs c1,c2] [--runs N] [--partition P] [--exp-name NAME] [--dataset-group short|long] [--sleep-sec S] [--submit-mode array|sequential] [--pipeline-mode two_stage|end_to_end] [--wandb-project PROJECT]

# Example: Adobe-2k, three configs, 5 runs each on polar4 (array-based two-stage)
bash submit_single_dataset.sh \
  --dataset-urls /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/safe-synthetics/cleaned/Adobe-2k.csv \
  --configs unsloth,dp,dp_usg_guidance \
  --runs 5 \
  --partition polar4 \
  --exp-name regex_adobe2k \
  --dataset-group short \
  --sleep-sec 5 \
  --submit-mode array \
  --pipeline-mode two_stage \
  --wandb-project regex_adobe2k

# Example: sequential per-run two-stage (GEN depends on its TRAIN)
bash submit_single_dataset.sh \
  --dataset-urls /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/safe-synthetics/cleaned/Adobe-2k.csv \
  --configs unsloth,dp \
  --runs 3 \
  --partition polar4 \
  --exp-name demo_exp \
  --dataset-group short \
  --sleep-sec 3 \
  --submit-mode sequential \
  --pipeline-mode two_stage

# Tail logs
tail -f /lustre/fsw/portfolios/llmservice/users/${USER_NAME}/nss_results/regex_adobe2k/short/*/slurm_*.out
```

Notes:
- The script honors time limits from `env_variables.sh` based on config name patterns (`unsloth`, `dp`, fallback `max`).
- Set `DATASET_GROUP` to `long` to use the long time limits.
- The dataset path is passed via `DATASET_URLS` and will be used directly by the runner.
- In `two_stage` mode, the TRAIN job creates a workdir at `--run-path` containing the adapter and config. The GEN job resumes from the same workdir and writes uniquely-timestamped output files (e.g., `synthetic_data_20260114T123456.csv`) allowing multiple generation runs from the same trained adapter.



### Monitoring and cancellation
```bash
squeue -u ${USER_NAME}
scancel <jobid>
```

### Collect results

Use W&B by setting `WANDB_MODE=online` in `env_variables.sh` and add your W&B token to `.api_tokens.sh`.

### Troubleshooting
- "USER_NAME is not set": run `export USER_NAME=...` and retry.
- Missing token file/key: create `${LUSTRE_DIR}/.api_tokens.sh` with `NIM_API_KEY` and `chmod 600`.
- Missing config files: verify `CONFIGS` in `env_variables.sh` and files in `CONFIG_DIR`.
- Permission errors: confirm your `/lustre/.../${USER_NAME}` paths and file perms.

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
At this time, the best recommendation is to place this in someone's user directory, so Kendrick created `/lustre/fsw/portfolios/llmservice/users/kendrickb/shared_safe_synthesizer` on the `cw-dfw-cs` and `cs-oci-ord` clusters.
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

From a file copier node on the new cluster, run the following to copy Kendrick's shared directory from dfw. Took ~30 minutes when run in Jan 2026 to copy 16 GB.

```
rsync -avzP cw-dfw-cs-001-dc-02.cw-dfw-cs-001.hpc.nvidia.com:/lustre/fsw/portfolios/llmservice/users/kendrickb/shared_safe_synthesizer/ /lustre/fsw/portfolios/llmservice/users/kendrickb/shared_safe_synthesizer/
```

Also good to check on ownership and permissions after copying to ensure 775 permissions (for directories) or 664 (for files).

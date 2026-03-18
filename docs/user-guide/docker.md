<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Running in Docker

Run the full Safe Synthesizer pipeline in a container with GPU access.
No local Python install required -- the container ships everything needed
for training, generation, and evaluation.

---

## Prerequisites

- Docker 20.10+ (BuildKit enabled by default in 23.0+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured
- NVIDIA driver compatible with CUDA 12.8
- NVIDIA GPU (A100 or better recommended)

Verify GPU access works:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi
```

---

## Quick Start

The container wraps the `safe-synthesizer` CLI. Mount your data and
Hugging Face cache, then pass CLI arguments after the image name:

```bash
docker run --gpus all --shm-size=1g \
  -v /path/to/your/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest \
  run --config /workspace/data/config.yaml --url /workspace/data/input.csv
```

The entrypoint prints helpful warnings if it detects common mistakes
(empty `/workspace`, missing `HF_HOME`, no GPU access).

More examples:

```bash
# Train only
docker run --gpus all --shm-size=1g \
  -v /path/to/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest run train --url /workspace/data/input.csv

# Generate from a trained adapter
docker run --gpus all --shm-size=1g \
  -v /path/to/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest run generate --url /workspace/data/input.csv --auto-discover-adapter

# Validate a config file (no GPU needed)
docker run \
  -v /path/to/data:/workspace/data \
  nss-gpu:latest config validate --config /workspace/data/config.yaml
```

---

## Mounting Your Data

The container starts with an empty `/workspace`. You bring your own data
by bind-mounting host directories with `-v`:

```bash
docker run --gpus all --shm-size=1g \
  -v /home/user/project:/workspace/data \
  ...
  nss-gpu:latest run --url /workspace/data/input.csv
```

Docker requires absolute paths for bind mounts. Relative paths like
`-v data:/workspace/data` are silently interpreted as named volumes --
Docker won't error, but you'll get an empty mount instead of your host
directory. Use `$(pwd)` to expand relative paths:

```bash
-v $(pwd)/my_data:/workspace/data    # correct
-v my_data:/workspace/data           # wrong -- Docker treats this as a named volume
```

You can mount multiple directories at different paths:

```bash
docker run --gpus all \
  -v /data/inputs:/workspace/inputs \
  -v /data/configs:/workspace/configs \
  -v /data/output:/workspace/output \
  -e NSS_ARTIFACTS_PATH=/workspace/output \
  ...
  nss-gpu:latest run --config /workspace/configs/my_config.yaml --url /workspace/inputs/data.csv
```

Artifacts are written to `/workspace/safe-synthesizer-artifacts/` by default
(override with `NSS_ARTIFACTS_PATH`). Make sure to mount a host directory
there if you want to retrieve results after the container exits.

---

## Secrets and API Keys

Pass secrets as environment variables at runtime -- never bake them into the
image. The most common ones:

```bash
docker run --gpus all --shm-size=1g \
  -v /path/to/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  -e HF_TOKEN="hf_..." \
  nss-gpu:latest run --url /workspace/data/input.csv
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | For gated models | Hugging Face token for downloading gated models (Llama, Mistral, etc.). Get one at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `NIM_API_KEY` | For PII classification | API key for the NIM endpoint used by PII column classification. Only needed when `NIM_ENDPOINT_URL` is set |
| `NIM_ENDPOINT_URL` | For PII classification | NIM/OpenAI-compatible endpoint URL for PII column classification |
| `WANDB_API_KEY` | For experiment tracking | WandB API key. Only needed when `--wandb-mode online` is used |

If `HF_TOKEN` is already stored in your HF cache (`~/.cache/huggingface/token`),
mounting the cache directory is sufficient -- the Hub library reads the token
file automatically.

See [Environment Variables](environment.md) for the full reference.

---

## Hugging Face Model Cache

Safe Synthesizer downloads models from Hugging Face Hub on first use.
Mount a host directory to persist downloads across container runs:

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  ...
```

| Host path | Container path | Env var | Purpose |
|-----------|---------------|---------|---------|
| `~/.cache/huggingface` | `/workspace/.hf_cache` | `HF_HOME` | Model weights, tokenizers, configs |

Without this mount, models are downloaded into the container's ephemeral
filesystem and lost when it exits.

For shared environments (team servers, CI), point at a shared cache:

```bash
-v /shared/hf_cache:/workspace/.hf_cache -e HF_HOME=/workspace/.hf_cache
```

---

## GPU Access

The image declares `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so the NVIDIA Container Toolkit
knows it needs GPU access. You still need `--gpus` to tell Docker to inject
the GPU devices:

```bash
# All GPUs
docker run --gpus all ...

# Specific GPUs
docker run --gpus '"device=0,1"' ...
```

To restrict which GPUs are visible inside the container, override the
environment variable:

```bash
docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=0,1 ...
```

---

## Shared Memory (`--shm-size`)

PyTorch uses `/dev/shm` for inter-process communication during training
(multi-worker data loading). Docker defaults to 64 MB, which causes
"Bus error" crashes. Always pass `--shm-size=1g` (or `--ipc=host`) when
running training workloads:

```bash
docker run --gpus all --shm-size=1g ...
```

The entrypoint script warns if `/dev/shm` is below 256 MB.
Generation-only runs are typically fine without it.

---

## File Permissions

The container runs as `appuser` (uid 1000). When bind-mounting host
directories, Docker preserves host ownership. If your host user has a
different uid, writes to the mounted directory (artifacts, outputs) will
fail with "Permission denied".

Fix by matching the container user to your host uid:

```bash
docker run --gpus all --user "$(id -u):$(id -g)" \
  -v /path/to/data:/workspace/data \
  ...
```

This overrides `appuser` with your host identity. The `--user` flag also
works with the dev image and interactive shells.

---

## Offline and Air-Gapped Environments

Pre-cache models by running the pipeline once with internet access, then
reuse the populated cache in the target environment:

```bash
# Step 1: populate cache (internet required)
docker run --gpus all --shm-size=1g \
  -v /path/to/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest run --config /workspace/data/config.yaml --url /workspace/data/input.csv

# Step 2: use in offline environment
docker run --gpus all --shm-size=1g \
  -v /path/to/data:/workspace/data \
  -v /shared/hf_cache:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  -e HF_HUB_OFFLINE=1 \
  nss-gpu:latest run --config /workspace/data/config.yaml --url /workspace/data/input.csv
```

See [Environment Variables -- Hugging Face Cache](environment.md#hugging-face-cache)
for details on `HF_HOME`, `HF_HUB_OFFLINE`, and `VLLM_CACHE_ROOT`.

---

## Building from Source

If pulling a pre-built image is not available, build locally:

```bash
make container-build-gpu           # runtime image
make container-build-gpu-dev       # dev image with test tooling
```

Override build arguments for different CUDA or Python versions:

```bash
docker build -f containers/Dockerfile.cuda \
  --build-arg CUDA_VERSION=12.6.3 \
  --build-arg PYTHON_VERSION=3.12.10 \
  --target runtime -t nss-gpu:custom .
```

See [Developer Guide -- Docker](../developer-guide/docker.md) for
build stages, ARG reference, and customization details.

---

## Interactive Shell

To explore the container or debug issues, override the entrypoint to get
a bash shell. Mount your data the same way as a normal run:

```bash
docker run -it --gpus all --shm-size=1g \
  -v $(pwd)/my_data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  --entrypoint /bin/bash \
  nss-gpu:latest
```

Inside the container you can run `safe-synthesizer` commands directly:

```bash
appuser@container:/workspace$ safe-synthesizer run --url /workspace/data/input.csv
appuser@container:/workspace$ safe-synthesizer config validate --config /workspace/data/config.yaml
```

---

## Makefile Shortcuts

For developers with the repo checked out, the Makefile provides convenience
targets that handle GPU flags, HF cache mounts, and workspace bind mounts:

| Command | What it does |
|---------|-------------|
| `make container-build-gpu` | Build the runtime image |
| `make container-run-gpu CMD="run --config ..."` | Run a pipeline command |
| `make container-build-gpu-dev` | Build the dev image |
| `make container-run-gpu-dev CMD="make test"` | Run a command in the dev container |

Override variables as needed:

```bash
make container-run-gpu CONTAINER_HF_CACHE=/shared/hf_cache CMD="run --url /workspace/data.csv"
```

Mount data from outside the repo tree with `CONTAINER_EXTRA_MOUNTS`:

```bash
make container-run-gpu \
  CONTAINER_EXTRA_MOUNTS="-v /data/sensitive:/workspace/data" \
  CMD="run --url /workspace/data/customers.csv"
```

---

## What to Read Next

- [Running Safe Synthesizer](running.md) -- pipeline execution, CLI commands
- [Configuration Reference](configuration.md) -- parameter tables
- [Environment Variables](environment.md) -- `HF_HOME`, `NSS_ARTIFACTS_PATH`, logging
- [Troubleshooting](troubleshooting.md) -- OOM fixes, offline errors

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

```bash
# Build the container (one-time, ~15 min on first build)
make container-build-gpu

# Run the full pipeline
docker run --gpus all \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest \
  run --config /workspace/config.yaml --url /workspace/data.csv
```

The container wraps the `safe-synthesizer` CLI. Arguments after the image
name are passed directly:

```bash
# Train only
docker run --gpus all -v $(pwd):/workspace nss-gpu:latest run train --url /workspace/data.csv

# Generate from a trained adapter
docker run --gpus all -v $(pwd):/workspace nss-gpu:latest \
  run generate --url /workspace/data.csv --auto-discover-adapter

# Validate a config file (no GPU needed)
docker run -v $(pwd):/workspace nss-gpu:latest config validate --config /workspace/config.yaml
```

---

## Volume Mounts

Bind-mount host directories to make data, config, and caches available
inside the container.

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| Your data directory | `/workspace` | Input data, config files, output artifacts |
| `~/.cache/huggingface` | `/workspace/.hf_cache` | Model downloads (persists across runs) |

Set `HF_HOME` inside the container to point at the mounted cache:

```bash
docker run --gpus all \
  -v /path/to/data:/workspace \
  -v /shared/hf_cache:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest run --config /workspace/config.yaml --url /workspace/input.csv
```

Artifacts are written to `/workspace/safe-synthesizer-artifacts/` by default.
Mount a host directory at `/workspace` to retrieve them after the run.

---

## GPU Access

Docker uses the `--gpus` flag to expose NVIDIA GPUs:

```bash
# All GPUs
docker run --gpus all ...

# Specific GPUs
docker run --gpus '"device=0,1"' ...
```

Alternatively, set `NVIDIA_VISIBLE_DEVICES`:

```bash
docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=0,1 ...
```

---

## Offline and Air-Gapped Environments

Pre-cache models by running the pipeline once with internet access, then
reuse the populated cache in the target environment:

```bash
# Step 1: populate cache (internet required)
docker run --gpus all \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest run --config /workspace/config.yaml --url /workspace/data.csv

# Step 2: use in offline environment
docker run --gpus all \
  -v /shared/hf_cache:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  -e HF_HUB_OFFLINE=1 \
  nss-gpu:latest run --config /workspace/config.yaml --url /workspace/data.csv
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

## Makefile Shortcuts

| Command | What it does |
|---------|-------------|
| `make container-build-gpu` | Build the runtime image |
| `make container-run-gpu CMD="run --config ..."` | Run a pipeline command |
| `make container-shell-gpu` | Interactive bash in the runtime container |
| `make container-build-gpu-dev` | Build the dev image |
| `make container-shell-gpu-dev` | Interactive bash in the dev container |

The Makefile handles GPU flags, HF cache mounts, and workspace bind mounts.
Override variables as needed:

```bash
make container-run-gpu CONTAINER_HF_CACHE=/shared/hf_cache CMD="run --url /workspace/data.csv"
```

---

## What to Read Next

- [Running Safe Synthesizer](running.md) -- pipeline execution, CLI commands
- [Configuration Reference](configuration.md) -- parameter tables
- [Environment Variables](environment.md) -- `HF_HOME`, `NSS_ARTIFACTS_PATH`, logging
- [Troubleshooting](troubleshooting.md) -- OOM fixes, offline errors

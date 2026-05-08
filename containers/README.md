<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Container Images

Dockerfiles for running and testing Safe-Synthesizer in containers.

## Files

| File | Base | Purpose |
|------|------|---------|
| `Dockerfile.cuda` | `nvidia/cuda:12.8.1-runtime-ubuntu22.04` | GPU runtime and dev images for training, generation, and evaluation |
| `Dockerfile.test_ci` | `python:3.11-slim` | CPU-only test image (`mise run test:ci-container`) |
| `entrypoint.sh` | -- | Wrapper entrypoint for the runtime image (mount/GPU checks) |

## CUDA Image

Three build stages, selected via `--target`:

- `runtime` -- minimal image wrapping the `safe-synthesizer` CLI via `tini` + `entrypoint.sh`. Runs as non-root `appuser`. The entrypoint detects common mistakes (empty workspace, missing HF cache, no HF token, no GPU, low `/dev/shm`) and prints hints before delegating to `safe-synthesizer`.
- `dev` -- extends runtime with git, uv, pytest, and the full dev dependency group. Runs as root for flexibility.
- `deps` -- intermediate stage (not a useful target on its own).

### Quick Start

```bash
# Build the runtime image
mise run container:build:gpu

# Run with your data -- mount your dataset and HF model cache
docker run --gpus all --shm-size=1g \
  -v <full_path_to_data_folder>:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest \
  run --data-source /workspace/data/input.csv


# Run with a test dataset
docker run --gpus all --shm-size=1g \
  -v $(pwd)/tests/stub_datasets:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest \
  run --data-source /workspace/data/clinc_oos.csv

# Run the full pipeline with a config file
docker run --gpus all --shm-size=1g \
  -v $(pwd)/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest \
  run --config /workspace/data/config.yaml --data-source /workspace/data/input.csv

# Interactive shell (mount your data, override entrypoint)
docker run -it --gpus all --shm-size=1g \
  -v $(pwd)/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  --entrypoint /bin/bash \
  nss-gpu:latest

# Dev image with test tooling
mise run container:build:gpu-dev
CMD="mise run test" mise run container:run:gpu-dev
```

Key flags:

- `--gpus all` -- expose NVIDIA GPUs (requires nvidia-container-toolkit)
- `--shm-size=1g` -- increase `/dev/shm` for PyTorch training (default 64 MB causes "Bus error")
- `-v HOST:CONTAINER` -- bind-mount data and HF cache; Docker requires absolute paths (use `$(pwd)` to expand relative ones)
- `-e HF_HOME=...` -- persist model downloads across container runs
- `-e HF_TOKEN=...` -- Hugging Face token for gated models (Llama, Mistral, etc.)
- `-e NSS_INFERENCE_KEY=...` -- inference API key for PII column classification (optional; `NSS_INFERENCE_ENDPOINT` defaults to the NVIDIA integrate URL if unset)
- `-e WANDB_API_KEY=...` -- WandB API key for experiment tracking (optional)
- `--user "$(id -u):$(id -g)"` -- match host uid if you get "Permission denied" writing artifacts

### Build Arguments

| ARG | Default | Description |
|-----|---------|-------------|
| `CUDA_VERSION` | `12.8.1` | CUDA toolkit version in the base image tag |
| `UBUNTU_VERSION` | `22.04` | Ubuntu version in the base image tag |
| `CUDA_IMAGE_TYPE` | `runtime` | Base image variant (`runtime` or `devel`) |
| `PYTHON_VERSION` | `3.11.13` | Python version installed via `uv python install` to `/opt/python` |
| `UV_VERSION` | `0.9.30` | uv version for the deps stage (matches `.mise.toml` pin) |
| `TARGETARCH` | _(set by BuildKit)_ | Target architecture (`amd64` or `arm64`) |
| `CUDA_ARCH_FLAGS` | `80;86;90;90a` | CUDA SM capabilities for `nvcc` (override for arm64: `90;90a;120;120a`) |

Override at build time:

```bash
docker build -f containers/Dockerfile.cuda \
  --build-arg PYTHON_VERSION=3.12.10 \
  --build-arg CUDA_VERSION=12.6.3 \
  --target runtime -t nss-gpu:custom .
```

### Mise Tasks

| Task | Description |
|--------|-------------|
| `container:build:gpu` | Build the runtime image |
| `container:build:gpu-dev` | Build the dev image |
| `container:build:gpu-multiarch` | Build multi-arch manifest (requires `CONTAINER_GPU_REGISTRY`) |
| `container:run:gpu` | Run a command in the runtime container |
| `container:run:gpu-dev` | Run a command in the dev container |

See `mise tasks` for the full task list with usage hints.

## Multi-Architecture

The CUDA image supports `linux/amd64` and `linux/arm64` (Grace/Blackwell).

### Single-arch builds

Override `CONTAINER_GPU_PLATFORM` to build for a specific architecture:

```bash
CONTAINER_GPU_PLATFORM=linux/arm64 mise run container:build:gpu
```

### Multi-arch manifest

Building a multi-platform manifest requires `docker buildx` and a registry
to push to -- `--load` only works for single-platform images:

```bash
CONTAINER_GPU_REGISTRY=ghcr.io/nvidia-nemo mise run container:build:gpu-multiarch
```

This pushes a manifest containing both `amd64` and `arm64` images to the
registry as `ghcr.io/nvidia-nemo/nss-gpu:latest`.

### CUDA compute capabilities

When `CUDA_IMAGE_TYPE=devel` is used and kernels must be compiled, set
`CUDA_ARCH_FLAGS` to the appropriate SM values:

| Architecture | `CUDA_ARCH_FLAGS` | GPUs |
|--------------|-------------------|------|
| amd64 | `80;86;90;90a` | A100, A10/3090, H100 |
| arm64 | `90;90a;120;120a` | H100 Grace, Blackwell |

```bash
docker build -f containers/Dockerfile.cuda \
  --build-arg CUDA_IMAGE_TYPE=devel \
  --build-arg CUDA_ARCH_FLAGS="90;90a;120;120a" \
  --platform linux/arm64 \
  --target runtime -t nss-gpu:arm64 .
```

## CPU Test Image

`Dockerfile.test_ci` provides a CPU-only image for running unit tests locally
or in CI without a GPU. It uses a two-stage build: `setup` (system packages +
mise-managed tools) and `install-deps` (Python environment via
`mise run bootstrap-nss cpu`).

### Quick Start

```bash
# Run CI unit tests in a container
mise run test:ci-container

# Verify mise-managed tools install correctly (fast -- setup stage only)
mise run test:tool-install
```

### Mise Tasks

| Task | Description |
|--------|-------------|
| `container:build:test` | Build the full CPU test image (both stages) |
| `container:build:test-setup` | Build only the setup stage (tools, no Python deps) |
| `test:ci-container` | Build and run CI unit tests |
| `test:tool-install` | Verify mise-managed tools install correctly (setup stage only) |

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Container Images

Dockerfiles for running and testing Safe-Synthesizer in containers.

## Files

| File | Base | Purpose |
|------|------|---------|
| `Dockerfile.cuda` | `python:3.13-slim-bookworm` | GPU runtime and dev images; CUDA libraries come from the selected package extra |
| `Dockerfile.test_ci` | `python:3.13-slim` | CPU-only test image (`mise run test:ci-container`) |
| `entrypoint.sh` | -- | Wrapper entrypoint for the runtime image (mount/GPU checks) |

## GPU Image

`Dockerfile.cuda` is parameterized by the project extra:

| Variant | Extra | Status |
|---------|-------|--------|
| `cu129` | `cu129` | Built today |
| `cu130` | `cu130` | Add after the CUDA 13.0 dependency extra lands |

The image does not use `nvidia/cuda` as a base. PyTorch, vLLM, FlashInfer,
and the NVIDIA CUDA runtime libraries are installed from the locked Python
dependencies for the selected extra.

Two image targets and one helper stage are available:

- `uv` -- helper stage that provides pinned `uv` binaries from the official image.
- `runtime` -- slim CLI image that installs dependencies in layered sync steps, uses `tini` + `entrypoint.sh`, and runs as non-root `appuser`. The entrypoint detects common mistakes (empty workspace, missing HF cache, no HF token, no GPU, low `/dev/shm`) and prints hints before delegating to `safe-synthesizer`.
- `dev` -- extends runtime with `uv`, `make`, build tools, and the Python dev/test dependency group.

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

# Dev image with test tooling
mise run container:build:gpu-dev
CMD="mise run test" mise run container:run:gpu-dev
```

Key flags:

- `--gpus all` -- expose NVIDIA GPUs (requires nvidia-container-toolkit)
- `--shm-size=1g` -- increase `/dev/shm` for PyTorch training
- `-v HOST:CONTAINER` -- bind-mount data and HF cache; Docker requires absolute paths
- `-e HF_HOME=...` -- persist model downloads across container runs
- `-e HF_TOKEN=...` -- Hugging Face token for gated models
- `-e NSS_INFERENCE_KEY=...` -- inference API key for PII column classification

### Build Arguments

| ARG | Default | Description |
|-----|---------|-------------|
| `CONTAINER_EXTRA` | `cu129` | Python extra to install with `engine` |
| `CONTAINER_VARIANT` | `cu129` | Image variant label/tag suffix |
| `PACKAGE_VERSION` | unset | Optional PEP 440 version passed to `uv-dynamic-versioning` |
| `PYTHON_VERSION` | `3.13` | Python slim image version |
| `PYTHON_IMAGE` | `python:${PYTHON_VERSION}-slim-bookworm` | Runtime and dev base image |
| `UV_IMAGE` | `ghcr.io/astral-sh/uv:0.9.30` | Source image for pinned `uv` binaries |

Override at build time:

```bash
docker build -f containers/Dockerfile.cuda \
  --build-arg CONTAINER_EXTRA=cu129 \
  --build-arg CONTAINER_VARIANT=cu129 \
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

Useful overrides:

```bash
CONTAINER_GPU_EXTRA=cu129 \
CONTAINER_GPU_VARIANT=cu129 \
CONTAINER_GPU_IMAGE=nss-gpu-cu129:latest \
  mise run container:build:gpu
```

## Multi-Architecture

The Dockerfile accepts `--platform` through Docker/Buildx. The default local
target is `linux/amd64`:

```bash
CONTAINER_GPU_PLATFORM=linux/arm64 mise run container:build:gpu
```

Multi-platform manifests must be pushed to a registry:

```bash
CONTAINER_GPU_REGISTRY=registry.example.com/team \
CONTAINER_GPU_IMAGE=safe-synthesizer:custom-cu129 \
  mise run container:build:gpu-multiarch
```

This builds and pushes `$(CONTAINER_GPU_REGISTRY)/$(CONTAINER_GPU_IMAGE)`.
Confirm the selected Python extra has wheels for every requested architecture
before enabling a platform in CI.

## CPU Test Image

`Dockerfile.test_ci` provides a CPU-only image for running unit tests locally
or in CI without a GPU. Its `setup` stage installs system packages and
mise-managed tools, while `install-deps` creates the Python environment with
`mise run bootstrap-nss cpu`. The separate `wheel-install` stage installs the
built wheel without project sources, runs CPU package and CLI checks, and
resolves the CUDA dependency set.

```bash
# Run CI unit tests in a container
mise run test:ci-container

# Verify mise-managed tools install correctly (fast -- setup stage only)
mise run test:tool-install

# Verify the built wheel in a clean container stage
mise run build-wheel
mise run release:verify-wheel
```

### CPU Test Mise Tasks

| Task | Description |
|--------|-------------|
| `container:build:test` | Build the full CPU test image (both stages) |
| `container:build:test-setup` | Build only the setup stage (tools, no Python deps) |
| `test:ci-container` | Build and run CI unit tests |
| `test:tool-install` | Verify mise-managed tools install correctly (setup stage only) |
| `release:verify-wheel` | Install and verify the built wheel in the clean wheel stage |

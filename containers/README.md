<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Container Images

Dockerfiles for running and testing Safe-Synthesizer in containers.

## Files

| File | Base | Purpose |
|------|------|---------|
| `Dockerfile.cuda` | `nvidia/cuda:12.8.1-runtime-ubuntu22.04` | GPU runtime and dev images for training, generation, and evaluation |
| `Dockerfile.test_ci` | `python:3.11-slim` | CPU-only image for running unit tests locally (`make test-ci-container`) |

## CUDA Image

Three build stages, selected via `--target`:

- `runtime` -- minimal image wrapping the `safe-synthesizer` CLI. Runs as non-root `appuser`.
- `dev` -- extends runtime with git, make, uv, pytest, and the full dev dependency group. Runs as root for flexibility.
- `deps` -- intermediate stage (not a useful target on its own).

### Quick Start

```bash
# Build the runtime image
make container-build-gpu

# Run the pipeline
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v ~/.cache/huggingface:/workspace/.hf_cache \
  -e HF_HOME=/workspace/.hf_cache \
  nss-gpu:latest run --config /workspace/data/config.yaml --url /workspace/data/input.csv

# Interactive shell
make container-shell-gpu

# Dev image with test tooling
make container-build-gpu-dev
make container-shell-gpu-dev
```

### Build Arguments

| ARG | Default | Description |
|-----|---------|-------------|
| `CUDA_VERSION` | `12.8.1` | CUDA toolkit version in the base image tag |
| `UBUNTU_VERSION` | `22.04` | Ubuntu version in the base image tag |
| `CUDA_IMAGE_TYPE` | `runtime` | Base image variant (`runtime` or `devel`) |
| `PYTHON_VERSION` | `3.11.13` | Python version installed via `uv python install` to `/opt/python` |
| `UV_VERSION` | `0.9.14` | uv version (matches `pyproject.toml` lower bound) |

Override at build time:

```bash
docker build -f containers/Dockerfile.cuda \
  --build-arg PYTHON_VERSION=3.12.10 \
  --build-arg CUDA_VERSION=12.6.3 \
  --target runtime -t nss-gpu:custom .
```

### Makefile Targets

| Target | Description |
|--------|-------------|
| `container-build-gpu` | Build the runtime image |
| `container-build-gpu-dev` | Build the dev image |
| `container-run-gpu` | Run a command in the runtime container |
| `container-shell-gpu` | Interactive bash in the runtime container |
| `container-run-gpu-dev` | Run a command in the dev container |
| `container-shell-gpu-dev` | Interactive bash in the dev container |

See `make help` for the full target list with usage hints.

## CI Test Image

Used by `make test-ci-container` for running unit tests in a Linux container
(useful on macOS where the native environment differs). Not intended for GPU
workloads.

```bash
make test-ci-container
```

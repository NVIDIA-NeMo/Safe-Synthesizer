<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dockerfile Patterns

Multi-stage builds, uv integration, CUDA, base images, and layer ordering for Python/ML projects.

## CPU Multi-Stage Build

Builder installs deps with uv, runtime copies only the venv.

```dockerfile
FROM ghcr.io/astral-sh/uv:0.9.14 AS uv

FROM python:3.11-slim-bookworm AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_INSTALLER_METADATA=true
WORKDIR /app

# Deps first (cached if lockfile unchanged)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# Then source (invalidates only this layer on code change)
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM python:3.11-slim-bookworm AS runtime
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder --chown=appuser:root /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
USER appuser
```

## GPU Multi-Stage Build

For services that need CUDA at runtime (inference, vLLM, etc.).

```dockerfile
FROM ghcr.io/astral-sh/uv:0.9.14 AS uv

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS builder
# Install Python (not included in CUDA images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder --chown=appuser:root /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER appuser
```

## GPU Layer Ordering

For large GPU images, order layers by size and stability (largest/most-stable first) to maximize cache reuse:

```
Layer 1: CUDA runtime libs     (~4.3 GB)  -- changes rarely
Layer 2: PyTorch               (~2.0 GB)  -- changes with torch version bumps
Layer 3: Inference engines      (~3.4 GB)  -- vLLM, Triton, xformers
Layer 4: Application deps       (~3.0 GB)  -- changes most often
```

Use explicit `COPY` from site-packages to control layer boundaries:

```dockerfile
# Copy CUDA-related packages as one layer
RUN --mount=from=builder,source=/app/.venv/lib/python3.11/site-packages,target=/tmp/sp \
    tar -cf - -C /tmp/sp nvidia torch triton | tar -xf - -C /app/.venv/lib/python3.11/site-packages/

# Copy everything else as a second layer
RUN --mount=from=builder,source=/app/.venv/lib/python3.11/site-packages,target=/tmp/sp \
    tar -cf - -C /tmp/sp --exclude=nvidia --exclude=torch --exclude=triton . \
    | tar -xf - -C /app/.venv/lib/python3.11/site-packages/
```

## CUDA Wheel Compilation

For building Python extensions that compile against CUDA (flash-attn, etc.). Requires `-devel-` images with headers and compilers.

```dockerfile
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.9.0+cu128
ARG BASE_IMAGE="nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04"

FROM ${BASE_IMAGE} AS setup_packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl g++ build-essential ca-certificates git

FROM setup_packages AS build_wheel
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/bin" sh
WORKDIR /build
ENV UV_COMPILE_BYTECODE=0 UV_NO_INSTALLER_METADATA=1 UV_LINK_MODE=copy UV_FROZEN=true
ENV MAX_JOBS=32

# Target specific GPU architectures
ENV CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80;\
-gencode arch=compute_86,code=sm_86;\
-gencode arch=compute_90,code=sm_90;\
-gencode arch=compute_90a,code=sm_90a;\
-gencode arch=compute_120,code=sm_120;\
-gencode arch=compute_120a,code=sm_120a"

RUN uv python install ${PYTHON_VERSION} && uv venv --python ${PYTHON_VERSION} --seed
ENV PATH="/build/.venv/bin:$PATH"
RUN uv pip install packaging psutil ninja numpy torch==${TORCH_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Clone, checkout, build
ARG PACKAGE_VERSION=2.8.4
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && git checkout v${PACKAGE_VERSION}
RUN cd flash-attention && uv build --wheel --no-build-isolation --out-dir /output

FROM scratch AS output
COPY --from=build_wheel /output/*.whl /
```

Build and extract wheels locally:

```bash
docker buildx build \
    --output type=local,dest=./wheels \
    -f Dockerfile.flash_attn \
    --platform linux/amd64 \
    --progress=plain .
```

## uv in Docker Reference

### Binary copy (preferred for reproducibility)

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /bin/uv
```

### Environment variables

```dockerfile
ENV UV_COMPILE_BYTECODE=1       # Pre-compile .pyc for faster startup
ENV UV_LINK_MODE=copy           # Copy files instead of hardlinks (works across layers)
ENV UV_NO_INSTALLER_METADATA=1  # Deterministic layers (no installer metadata)
```

For wheel builds, set `UV_COMPILE_BYTECODE=0` (bytecode not needed for build artifacts).

### Cache mount

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable
```

### Common flag combinations

```bash
# Service image: install all deps, no editable installs
uv sync --frozen --no-install-project --no-editable

# Then with source:
uv sync --frozen --no-editable

# Specific dependency group only:
uv sync --frozen --only-group gpu-tasks --no-editable

# Dev overlay on pre-built image:
uv pip install --no-deps --reinstall --no-cache .
```

## ONBUILD Base Images

For multi-service repos sharing a common base:

```dockerfile
# Base image (built once, reused by services)
FROM python:3.11-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

ONBUILD COPY --from=builder /app/.venv /app/.venv
ONBUILD ENV PATH="/app/.venv/bin:$PATH"
ONBUILD ENTRYPOINT ["python", "-m"]
ONBUILD CMD ["--help"]
```

Each service Dockerfile then just needs a `builder` stage and a `FROM base AS runtime`.

## Dockerfile.dev Pattern

Fast iteration by overlaying code on a pre-built image:

```dockerfile
ARG BASE_IMAGE=myregistry/myapp:latest
FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /bin/uv
COPY pyproject.toml uv.lock src/ ./
RUN uv pip install --no-deps --reinstall --no-cache .
```

## BuildKit 2025 Features

### COPY --exclude (GA)

```dockerfile
# Copy source without tests or docs
COPY --exclude=tests --exclude=docs --exclude=*.md . /app
```

### Image checksum pinning

```dockerfile
# Pin base image for reproducibility
FROM python:3.11-slim-bookworm@sha256:abc123...
```

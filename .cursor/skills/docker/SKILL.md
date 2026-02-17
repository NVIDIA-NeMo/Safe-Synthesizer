---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: docker
description: "Docker containerization for Python/uv/CUDA projects. Covers Dockerfiles, multi-stage builds, buildx bake, GPU images, image optimization, container security, and dev workflows. Trigger keywords - Dockerfile, container, Docker, image, buildx, bake, CUDA, GPU image, container security, .dockerignore, multi-stage, distroless, health check."
---

# Docker for Python/uv/CUDA

Production-grade Docker patterns for Python projects using uv and optional CUDA/GPU support.

## Repo Docker Commands

```bash
# Build the CI test image
make container-build-test

# Run CI tests in a Linux container (matches CI environment)
make test-ci-container
```

## Base Image Selection

| Image | Use Case | Size |
|-------|----------|------|
| `python:3.11-slim-bookworm` | CPU services, API, tasks | ~150MB |
| `nvidia/cuda:12.8-runtime-ubuntu22.04` | GPU runtime (inference, serving) | ~4GB |
| `nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` | GPU compilation (wheel builds, flash-attn) | ~8GB |
| Distroless Python | Production, minimal attack surface | ~120MB |
| `ghcr.io/astral-sh/uv:0.9.14` | uv binary source (copy into builder) | -- |

**Rule of thumb**: use `-runtime-` for running code, `-devel-` for compiling extensions.

## Quick Multi-Stage Pattern (CPU)

```dockerfile
FROM ghcr.io/astral-sh/uv:0.9.14 AS uv

FROM python:3.11-slim-bookworm AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM python:3.11-slim-bookworm AS runtime
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder --chown=appuser:root /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

## Quick CUDA Wheel Build Pattern

```dockerfile
ARG CUDA_VERSION=12.8.1
ARG BASE_IMAGE="nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04"

FROM ${BASE_IMAGE} AS builder
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl g++ build-essential git ca-certificates
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/bin" sh
WORKDIR /build
RUN uv python install 3.11 && uv venv --python 3.11 --seed
ENV PATH="/build/.venv/bin:$PATH" MAX_JOBS=32
# Install build deps, then build wheel
RUN uv pip install torch==2.9.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
COPY . .
RUN uv build --wheel --no-build-isolation --out-dir /output

FROM scratch AS output
COPY --from=builder /output/*.whl /
```

Build and extract: `docker buildx build --output type=local,dest=./wheels -f Dockerfile .`

## Reference Table

| Topic | Reference |
|-------|-----------|
| Multi-stage builds, uv, CUDA, base images | `./references/dockerfile-patterns.md` |
| docker-bake.hcl, buildx bake, registry cache | `./references/bake-orchestration.md` |
| Non-root, distroless, secrets, capabilities | `./references/security-hardening.md` |
| Image size, layer caching, .dockerignore | `./references/optimization.md` |
| Dockerfile.dev, compose, test containers | `./references/dev-workflow.md` |
| Symptom/cause/fix, code review checklist | `./references/diagnostics.md` |

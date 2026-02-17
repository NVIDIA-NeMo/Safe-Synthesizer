<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Image Optimization

Image size reduction, layer caching strategies, .dockerignore, and GPU-specific layer management.

## Base Image Size Comparison

| Base Image | Approx Size | Best For |
|------------|-------------|----------|
| `python:3.11` | ~900MB | Never use in production |
| `python:3.11-slim-bookworm` | ~150MB | CPU services, general use |
| `python:3.11-alpine` | ~50MB | Size-critical, if deps are compatible |
| Distroless Python | ~120MB | Max security, no shell |
| `nvidia/cuda:12.8-runtime-*` | ~4GB | GPU inference/serving |
| `nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-*` | ~8GB | Wheel compilation only |

**Alpine caveat**: Many Python packages with C extensions (numpy, pandas) need extra build tools on Alpine. Prefer `slim-bookworm` unless image size is critical.

## Layer Ordering

Order instructions from least-changing to most-changing:

```dockerfile
# 1. System deps (rarely change)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Create user (rarely changes)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 3. Dependency files (change occasionally)
COPY pyproject.toml uv.lock ./

# 4. Install deps (cached if lockfile unchanged)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 5. Source code (changes frequently)
COPY src/ src/
```

### Bad: source before deps

```dockerfile
# Every source change reinstalls all deps
COPY . .
RUN uv sync --frozen
```

## GPU Layer Strategy

For images with large ML dependencies, control layer boundaries explicitly:

```
Layer 1: CUDA runtime + cuDNN     ~4.3 GB  (changes: CUDA version bumps)
Layer 2: PyTorch                  ~2.0 GB  (changes: torch version bumps)
Layer 3: Inference engines        ~3.4 GB  (changes: vLLM/triton updates)
Layer 4: Application code + deps  ~0.5 GB  (changes: every commit)
```

When app code changes, only Layer 4 needs rebuilding. Layers 1-3 are cached.

## .dockerignore Template

```dockerignore
# Version control
.git
.gitignore

# Python artifacts
__pycache__
*.pyc
*.pyo
*.egg-info
dist/
build/
.eggs/

# Virtual environments
.venv
venv

# IDE
.vscode
.idea
*.swp

# Testing/coverage
.pytest_cache
htmlcov/
coverage.*
.ruff_cache

# Documentation
docs/
*.md
!README.md

# ML artifacts (large)
*.ckpt
*.pt
*.bin
*.safetensors
model_checkpoints/

# Docker (avoid recursive)
Dockerfile*
docker-compose*
.dockerignore

# Misc
.env
.env.local
.cursor
.agent
```

## Cache Mounts

### uv cache

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable
```

### apt cache

```dockerfile
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl
```

### pip cache (when not using uv)

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
```

## Clean Up in Same Layer

```dockerfile
# Good: install and clean in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Bad: cleanup in separate layer (doesn't reduce size)
RUN apt-get update && apt-get install -y curl
RUN rm -rf /var/lib/apt/lists/*
```

## BuildKit COPY --exclude

Skip files from the build context without .dockerignore (BuildKit 2025, GA):

```dockerfile
COPY --exclude=tests --exclude=docs --exclude=*.md . /app
```

## Image Analysis

```bash
# View layer sizes
docker history myimage:latest

# Detailed layer analysis (install dive first)
docker run --rm -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    wagoodman/dive:latest myimage:latest

# Print bake config without building
docker buildx bake --print
```

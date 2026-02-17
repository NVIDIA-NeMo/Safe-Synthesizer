<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security Hardening

Non-root users, distroless images, secrets management, capabilities, and health checks.

## Non-Root User

Always create and switch to a non-root user in production images.

### Debian/Ubuntu (slim-bookworm, CUDA)

```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder --chown=appuser:root /app/.venv /app/.venv
WORKDIR /app
USER appuser
```

### Alpine

```dockerfile
RUN addgroup -S appuser && adduser -S appuser -G appuser
USER appuser
```

### K8s / OpenShift Compatible

OpenShift runs containers with arbitrary UIDs in the `root` group. Use `--chmod=g=u` to ensure group-writable:

```dockerfile
COPY --from=builder --chown=appuser:root --chmod=g=u /app/.venv /app/.venv
```

## Distroless Runtime Images

Distroless images contain only the runtime (no shell, no package manager). Smallest attack surface.

```dockerfile
FROM python:3.11-slim-bookworm AS builder
# ... build steps ...

FROM gcr.io/distroless/python3-debian12 AS runtime
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
USER nonroot
```

**Trade-off**: no shell means no `exec` for debugging. Keep a `dev` target with a shell for development.

## BuildKit Secrets

Never bake secrets into image layers. Use BuildKit secret mounts:

### In Dockerfile

```dockerfile
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) && \
    huggingface-cli download mymodel --token $HF_TOKEN
```

### With docker build

```bash
docker build --secret id=HF_TOKEN,env=HF_TOKEN .
```

### With docker-bake.hcl

```hcl
target "gpu-tasks" {
    secret = [
        "type=env,id=HF_TOKEN",
        "type=env,id=NGC_CLI_API_KEY",
    ]
}
```

### Common secrets in ML projects

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | Hugging Face model downloads |
| `NGC_CLI_API_KEY` | NVIDIA NGC container/model registry |
| `WANDB_API_KEY` | Weights & Biases logging |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 model storage |

## Minimal Capabilities

In Docker Compose or runtime, drop all capabilities and add back only what's needed:

```yaml
services:
  app:
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if binding to port < 1024
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
```

## Image Pinning

Pin base images by digest for reproducible builds:

```dockerfile
# Pinned (reproducible)
FROM python:3.11-slim-bookworm@sha256:a1b2c3d4e5f6...

# Unpinned (may change)
FROM python:3.11-slim-bookworm
```

Get a digest:

```bash
docker pull python:3.11-slim-bookworm
docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim-bookworm
```

## Health Checks

### HTTP Service

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/ready || exit 1
```

### Without curl (smaller image)

```dockerfile
# Python
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# wget (Alpine)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1
```

### Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--interval` | 30s | Time between checks |
| `--timeout` | 10s | Max time for a check to complete |
| `--start-period` | 5-60s | Grace period for container startup (longer for ML model loading) |
| `--retries` | 3 | Failures before marking unhealthy |

For ML services that load large models, increase `--start-period` (e.g., 120s).

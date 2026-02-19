<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Diagnostics

Symptom/cause/fix table for common Docker issues and a code review checklist.

## Troubleshooting Table

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Build is slow (10+ min) | Poor layer ordering; deps reinstalled on every code change | Copy `pyproject.toml`/`uv.lock` before `src/`; add `--mount=type=cache` |
| Image is too large (1GB+ for CPU) | Build tools or dev deps in production image | Use multi-stage build; copy only `.venv` to runtime stage |
| GPU not visible in container | Missing NVIDIA runtime or container toolkit | Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/); use `--gpus all` |
| uv cache miss every build in CI | No cache mount or no registry cache | Add `--mount=type=cache,target=/root/.cache/uv`; add `cache-from`/`cache-to` in bake |
| Permission denied at runtime | App expects root but container runs as non-root | Add `RUN groupadd/useradd` + `COPY --chown` + `USER` |
| venv missing after bind mount | Mount hides the in-image venv | Set `UV_PROJECT_ENVIRONMENT=/opt/venv` (outside mount path) |
| CUDA version mismatch | Container CUDA version incompatible with host driver | Match CUDA runtime version to host driver's [compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) |
| Wheel build fails: missing headers | Using `-runtime-` image instead of `-devel-` | Use `nvcr.io/nvidia/cuda:*-cudnn-devel-*` for compilation |
| Wheel build OOM during compilation | Too many parallel compilation jobs | Reduce `MAX_JOBS` (e.g., 16 instead of 32); or add `--mount=type=tmpfs,target=/tmp` |
| `apt-get` fails with lock errors | Concurrent apt access in parallel builds | Add `sharing=locked` to apt cache mounts |
| Container starts but health check fails | App not ready; or wrong port/path | Increase `--start-period` (especially for ML model loading); verify endpoint |
| `COPY --exclude` not working | Older BuildKit version | Requires BuildKit with `# syntax=docker/dockerfile:1` or Docker Engine 28+ |
| Build context too large | No `.dockerignore`; sending GB of data/models | Add `.dockerignore` excluding `.git`, `.venv`, `*.ckpt`, `*.safetensors`, etc. |

## Code Review Checklist

When reviewing Dockerfiles, verify:

### Build Efficiency
- [ ] Multi-stage build separates builder and runtime
- [ ] Dependencies copied before source code (layer caching)
- [ ] Cache mounts for uv/pip/apt (`--mount=type=cache`)
- [ ] `.dockerignore` excludes large/unnecessary files
- [ ] No unnecessary packages installed

### Security
- [ ] Non-root user created and used (`USER appuser`)
- [ ] Files owned by non-root user (`--chown=appuser:root`)
- [ ] No secrets in `ENV`, `ARG`, or `COPY` (use `--mount=type=secret`)
- [ ] Base image pinned (tag or digest)
- [ ] Production image is minimal (slim/distroless, not full or devel)

### Runtime
- [ ] `HEALTHCHECK` defined for services
- [ ] `EXPOSE` matches actual listening port
- [ ] Environment variables documented (`ENV` with sensible defaults)
- [ ] Appropriate `--start-period` for ML model loading

### GPU (if applicable)
- [ ] Runtime image used for serving (not devel)
- [ ] Devel image used for compilation only
- [ ] `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES` set
- [ ] `CUDA_ARCH_FLAGS` targets required GPU architectures
- [ ] Large layers (CUDA, PyTorch) ordered for cache efficiency

### CI/CD
- [ ] Registry cache configured (`cache-from`/`cache-to`)
- [ ] Multi-platform builds where needed (CPU: amd64+arm64; GPU: amd64 only)
- [ ] Secrets passed via BuildKit, not build args
- [ ] Image tags follow convention (commit SHA, branch, semver)

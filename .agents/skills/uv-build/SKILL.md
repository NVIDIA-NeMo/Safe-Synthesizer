---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: uv-build
description: "uv package management, dependency groups, PyTorch index handling, hatch build system, and versioning for this repo. Triggers on: uv, uv sync, uv lock, uv add, uv build, dependency, pyproject.toml, extras, cpu, cu129, hatch, wheel, version, publish."
license: Apache-2.0
---

# uv and Build System

Package management with uv, extras for CPU/CUDA, hatch build, and dynamic versioning.

## Bootstrap Commands

```bash
# Full dev environment (tools + Python + CPU deps)
mise run setup && mise run bootstrap-nss cpu

# Pick a variant:
mise run bootstrap-nss dev       # dev tools only (no engine/torch)
mise run bootstrap-nss cpu       # + engine + CPU PyTorch
mise run bootstrap-nss cu129     # + engine + CUDA 12.9 PyTorch
mise run bootstrap-nss cuda      # alias for cu129
mise run bootstrap-nss engine    # + engine (no torch)

# Slurm: force Python, caches, and the project venv onto Lustre
LUSTRE_DIR="/path/to/container-visible/project/directory" \
  MISE_IGNORED_CONFIG_PATHS="$HOME/.config/mise/config.toml" \
  MISE_LOCKED=1 mise run bootstrap-nss-slurm cu129
```

Under the hood: `uv sync --frozen --extra <extra> [--extra engine] --group dev`

`bootstrap-nss-slurm` requires `LUSTRE_DIR`, installs the pinned Python under
that directory, recreates `.venv` if its interpreter is not container-visible,
then runs the same frozen profile sync as `bootstrap-nss`.

## Extras and Conflicts

| Extra | What it installs |
|-------|------------------|
| `cpu` | PyTorch CPU, faiss-cpu, flashinfer (Linux only) |
| `cu129` | PyTorch+CUDA 12.9, faiss-gpu, flashinfer-jit-cache |
| `engine` | ML pipeline deps (outlines, wandb, tiktoken, etc.) -- no torch |
| `microservices` | `nemo-microservices` from local path |

`cpu` and `cu129` conflict -- you must pick one, never both. Enforced in `[tool.uv] conflicts`.

## Index Management

PyTorch wheels come from dedicated indexes, not PyPI:

| Index | URL | Used for |
|-------|-----|----------|
| `pytorch-cpu` | `download.pytorch.org/whl/cpu` | torch, torchvision (CPU, Linux) |
| `pytorch-cu129` | `download.pytorch.org/whl/cu129` | torch, torchvision, triton (CUDA) |
| `nv-shared-pypi-local` | NVIDIA Artifactory | Internal NVIDIA packages |
| `flashinfer-jit-cache-cu129` | `flashinfer.ai/whl/cu129` | FlashInfer JIT cache |
| `nvidia-pypi-public` | `pypi.nvidia.com` | Public NVIDIA packages |

All indexes are `explicit = true` (only used when a package is mapped to them in `[tool.uv.sources]`).

## Adding Dependencies

```bash
# Add to base dependencies
uv add <package>

# Add to a dependency group
uv add --group dev <package>
uv add --group test <package>

# Change CPU or CUDA runtime extras
# Edit cuda_deps.toml, regenerate pyproject.toml, then lock.
uv run --frozen tools/gen_cuda_deps.py cuda_deps.toml --pyproject pyproject.toml
uv lock
```

After any change: `uv lock` to regenerate `uv.lock`. Pre-commit verifies the lock is up to date.
The generated CPU/CUDA sections of `pyproject.toml` must not be edited directly;
`mise run lock-check` verifies that they match `cuda_deps.toml`. The
generator owns the complete `[tool.uv.sources]` and `[[tool.uv.index]]`
sections, so add every source or index there through `cuda_deps.toml`.

## Dependency Groups

| Group | Contains |
|-------|----------|
| `dev` | Includes `docs` + `test` groups, plus ipywidgets, pandas-stubs, prek, typer, etc. |
| `test` | pytest, pytest-asyncio, pytest-cov, pytest-env, pytest-subtests, pytest-timeout, pytest-xdist |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-gen-files, etc. |

## Running Tools

Always use `uv run` to ensure the correct environment:

```bash
uv run pytest ...
uv run --frozen pytest ...     # Don't update lock
uv run --group docs mkdocs serve
uv run --frozen --no-project --group docs mkdocs build
```

## Build and Version

```bash
# Build wheel (version from git tag via uv-dynamic-versioning)
mise run build-wheel     # or: uv build --wheel

# Publish to NVIDIA Artifactory
mise run publish:internal
```

Version source: `uv-dynamic-versioning` reads git tags (PEP 440 style). Fallback `0.0.0` for shallow clones.

Build backend: `hatchling` with wheel target `packages = ["src/nemo_safe_synthesizer"]`.

## Key pyproject.toml Sections

| Section | Purpose |
|---------|---------|
| `[tool.uv]` | Required version, cache-keys, conflicts, overrides, environments |
| `[tool.uv.sources]` | Map packages to specific indexes by extra/marker |
| `[[tool.uv.index]]` | Define named package indexes |
| `[build-system]` | hatchling + uv-dynamic-versioning |
| `[tool.hatch.version]` | Source: uv-dynamic-versioning |
| `[tool.uv-dynamic-versioning]` | Git VCS, PEP 440, fallback version |
| `[tool.vendor-package]` | Vendoring into NMP SDK |

## Vendor Package

`[tool.vendor-package]` configures vendoring Safe-Synthesizer into the NMP SDK:
- Target: `beta.safe_synthesizer`
- Includes specific paths from `src/` and `tests/`
- Used by the `prek` tool during NMP sync

## Conventions

1. Never use `pip` -- always `uv`
2. Use `--frozen` in CI and Make targets to prevent lock updates
3. Use `uv run` to run tools (pytest, mkdocs, etc.)
4. uv version is pinned in `.mise.toml`.
5. Edit non-generated `pyproject.toml` sections directly (e.g. dependency groups); CPU/CUDA extras go through `cuda_deps.toml` instead, then `uv lock`
6. Use `uv add` for base/group deps

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ty Configuration and Usage

Configuration and workflow for [ty](https://github.com/astral-sh/ty), the type checker from Astral (makers of Ruff/uv).

## Running ty

```bash
# Check the full project (respects [tool.ty.src] in pyproject.toml)
ty check

# Check specific files or directories
ty check src/nemo_safe_synthesizer/config/
ty check src/nemo_safe_synthesizer/config/model.py

# If ty isn't installed locally, run via uvx
uvx ty check
```

## Configuration

ty is configured in the repo's `pyproject.toml` under `[tool.ty.src]`.
See `pyproject.toml` at the repo root for the current exclusion list and settings.

### Available Options

```toml
[tool.ty]
# Python version to target
python-version = "3.11"

[tool.ty.src]
# Root directory for source files
root = "."

# Directories to exclude from type checking
exclude = [
    ".venv/",
    "__pycache__/",
]
```

## Repo-Specific Workflow

### Running on Changed Files Only

The repo provides `tools/lint/run-ty-check.sh` which runs ty only on files changed relative to `main` (or a given SHA):

```bash
# Check files changed since main
bash tools/lint/run-ty-check.sh

# Check files changed since a specific commit
bash tools/lint/run-ty-check.sh abc123
```

This script:
1. Gets the list of changed `.py` files via `git diff`
2. Filters out files matching `[tool.ty.src].exclude` using `tools/lint/filter_ty_exclusions.py`
3. Runs `ty check` on the remaining files

### Installation

ty is managed via the repo's `tools/binaries/tools.yaml`. Or install directly:

```bash
# Via uvx (no install needed)
uvx ty check

# Via uv tool install
uv tool install ty

# Via pip
pip install ty
```

## Inline Suppression

```python
# Suppress a specific ty error on a line
x: int = "hello"  # ty: ignore[invalid-assignment]

# Suppress all ty errors on a line
x: int = "hello"  # ty: ignore
```

## Comparing to mypy/pyright

| Feature | ty | mypy | pyright |
|---------|-----|------|---------|
| Speed | Very fast (Rust) | Slow | Fast |
| Config | `pyproject.toml` | `pyproject.toml` / `mypy.ini` | `pyrightconfig.json` / `pyproject.toml` |
| Suppression | `# ty: ignore` | `# type: ignore` | `# pyright: ignore` |
| Plugin system | No | Yes | No |

## Quick Reference

| Command | Description |
|---------|-------------|
| `ty check` | Check entire project |
| `ty check <path>` | Check specific path |
| `ty --version` | Show version |
| `uvx ty check` | Run without installing |
| `bash tools/lint/run-ty-check.sh` | Check changed files only |

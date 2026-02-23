---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: diagnose-deps
description: Diagnose transitive dependency changes by diffing uv.lock between git refs. Use when a test breaks after a lockfile update, an import error appears after dependency changes, CI fails with version mismatches, or the user asks what changed in dependencies. Trigger keywords - dependency, lockfile, uv.lock, transitive, dep change, version bump, package upgrade, package downgrade, bisect deps.
---

# Diagnose Dependency Changes

Use `tools/diff-lockfile.py` to compare `uv.lock` between git refs and surface every added, removed, upgraded, or downgraded package.

## Shell Permissions

All commands require `required_permissions: ["all"]`. The script uses PEP 723 inline dependencies that auto-install on first run (needs network), and git operations may need filesystem access.

## Quick Diff

```bash
# Auto merge-base with origin/main vs HEAD
uv run tools/diff-lockfile.py

# Explicit base ref
uv run tools/diff-lockfile.py origin/main

# Between two arbitrary commits
uv run tools/diff-lockfile.py abc123 --head def456
```

## JSON Output with jq

Use `--json` for programmatic analysis. Output is a flat JSON array of `PackageChange` objects.

```bash
# Count changes by category
uv run tools/diff-lockfile.py --json | jq '[group_by(.change)[] | {type: .[0].change, count: length}]'

# Check if a specific package changed
uv run tools/diff-lockfile.py --json | jq '.[] | select(.name == "transformers")'

# Filter GPU/CUDA-related changes
uv run tools/diff-lockfile.py --json | jq '[.[] | select(.name | test("torch|cuda|nvidia|flashinfer|vllm"))]'

# Find major version bumps (most likely to break)
uv run tools/diff-lockfile.py --json | jq '[.[] | select(.change == "upgraded") | select((.old.version | split(".")[0]) != (.new.version | split(".")[0]))]'

# Extract upgraded package specs for manual testing
uv run tools/diff-lockfile.py --json | jq -r '.[] | select(.change == "upgraded") | "\(.name)==\(.new.version)"'
```

## Diagnosis Decision Tree

1. Test broke after `uv lock`?
   Run the diff, look for upgraded packages in the failing test's dependency chain.

2. Import error?
   Filter JSON for added/removed packages matching the missing module:
   ```bash
   uv run tools/diff-lockfile.py --json | jq '.[] | select(.change == "removed" or .change == "added")'
   ```

3. Performance regression?
   Check if `torch`, `transformers`, `vllm`, or other heavy deps changed version.

4. Multiple PyTorch sources?
   `torch` appears 3x in the lockfile (PyPI, `+cpu`, `+cu128`). Filter with:
   ```bash
   uv run tools/diff-lockfile.py --json | jq '[.[] | select(.name == "torch")]'
   ```

5. Override conflict?
   Check `pyproject.toml` `override-dependencies` after seeing `outlines_core` or `flashinfer-*` in the diff. Transitive upgrades can break forced overrides.

## Fragile Dependency Chains

Packages most likely to cause breakage in this repo:

| Package | Why it breaks | Linked to |
|---------|--------------|-----------|
| `unsloth` | Requires exact `datasets==4.3.0`; invasive runtime patching | `datasets`, `torch` |
| `torch` (3 variants) | PyPI / `+cpu` / `+cu128` from different indexes | `torchvision`, `torchao`, `xformers` |
| `vllm` / `outlines_core` | Override to `0.2.13` conflicts with vllm's pin | `outlines` |
| `flashinfer-*` | 3 sub-packages with platform markers | CUDA toolkit |
| `transformers` / `peft` / `trl` | HuggingFace ecosystem; frequent coupled updates | `accelerate`, `torch` |

Dep-sensitive test areas: `tests/e2e/`, `tests/training/`, `tests/generation/` (import torch/transformers/vllm directly).

## Error Scenarios

| Problem | Cause | Fix |
|---------|-------|-----|
| "could not determine merge-base" | No `origin/main` remote | Use explicit base ref |
| Script slow on first run | PEP 723 deps downloading | One-time; subsequent runs are fast |
| "Lockfile not found at ref" | `uv.lock` didn't exist at that commit | Script exits 0; use a newer base ref |
| Bisect keeps skipping (exit 125) | `uv sync --frozen` failing | Lockfile invalid at that commit |

## Reference

`ChangeType` values: `added`, `removed`, `upgraded`, `downgraded`

`PackageChange` JSON schema:
```json
{
  "name": "numpy",
  "change": "upgraded",
  "old": {"name": "numpy", "version": "1.26.0", "source": "https://pypi.org/simple"},
  "new": {"name": "numpy", "version": "2.0.1", "source": "https://pypi.org/simple"},
  "ref": "abc123..def456"
}
```

Fields: `name` (str), `change` (ChangeType), `old` (Package or null), `new` (Package or null), `ref` (str).

CLI options: `[BASE_REF]`, `--head`, `--lockfile`, `--json`, `--run TEST_TARGET`

For detailed step-by-step workflows (bisect, cross-skill PR diagnosis, pre-merge review), see [workflows.md](workflows.md).

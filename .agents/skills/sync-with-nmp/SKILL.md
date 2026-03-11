# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
name: sync-with-nmp
description: Synchronize code between the Safe-Synthesizer GitHub repo and the internal NMP GitLab repo. Use when the user wants to sync changes from an NMP merge request, push changes to NMP, pull the latest NMP code, or manage the bidirectional sync workflow. Trigger keywords - sync, synchronize, NMP, GitLab, merge request, MR, rsync, upstream, downstream, pull from NMP, push to NMP.
---

# Sync with NMP

Synchronize code between Safe-Synthesizer (GitHub, public) and NMP (GitLab, internal). The `nemo_safe_synthesizer` package lives inside NMP at `packages/nemo_safe_synthesizer/`. nmp is likely installed at `../aire/microservices/nmp`.

## Architecture

```
Safe-Synthesizer (GitHub)           NMP (GitLab)
├── src/                    ←→      packages/nemo_safe_synthesizer/src/
├── tests/                  ←→      packages/nemo_safe_synthesizer/tests/
├── design.md               ←→      packages/nemo_safe_synthesizer/design.md
├── script/                 ←→      packages/nemo_safe_synthesizer/script/
└── (metafiles)             ←→      packages/nemo_safe_synthesizer/(metafiles)
```

Primary sync directions:

- Safe-Synthesizer → NMP ("downstream"): Push OSS contributions back into the internal repo.
- NMP → Safe-Synthesizer ("upstream"): Pull merged NMP changes into the OSS repo.

## Prerequisites

- NMP repo cloned locally (default: `$HOME/dev/aire/microservices/nmp`)
- `rsync` and `jq` installed
- Working directory is the Safe-Synthesizer repo root

## Shell Permissions

Always use `required_permissions: ["all"]` when running sync commands. `rsync` crosses repo boundaries.

## Environment

The `NMP_REPO_PATH` env var must point to the NMP repo root. It defaults to `$HOME/dev/aire/microservices/nmp` in `tools/sync-from-mr.sh`, but the Makefile targets require it explicitly.

```bash
export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"
```

## Quick Reference

| Task | Command |
|------|---------|
| Sync from a merged NMP MR | `make synchronize-from-nmp-mr MR=<number>` |
| Sync all Python files from NMP | `make synchronize-py-files-from-nmp` |
| Sync all files from NMP | `make synchronize-from-nmp` |
| Sync Python files to NMP | `make synchronize-py-files-to-nmp` |
| Sync all files to NMP | `make synchronize-to-nmp` |

## Sync from NMP MR (Most Common)

When a merge request lands in NMP that touches `packages/nemo_safe_synthesizer/`, sync it to Safe-Synthesizer:

```bash
# Creates a branch if on main, then rsyncs src/ and tests/ from the MR's squash commit
make synchronize-from-nmp-mr MR=<mr-iid>
```

What this does:

1. If on `main`, creates branch `$USER/sync-<MR>-from-nmp`
2. Calls `tools/sync-from-mr.sh` which:
   - Fetches the MR's squash commit SHA via the GitLab API
   - Checks out that commit in the NMP repo
   - Copies changed `src/` and `tests/` files from the MR's squash commit
   - Reports files changed in the MR that are outside `src/` and `tests/` (need manual review)

After syncing, follow the post-sync workflow below.

## Full Sync from NMP

For a bulk sync of the entire package (not tied to a specific MR):

```bash
export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"

# Sync src/ and tests/
make synchronize-py-files-from-nmp

# Also sync metafiles (design.md, script/, docs, etc.)
make synchronize-metafiles-from-nmp

# Or both at once
make synchronize-from-nmp
```

Note: Metafile sync excludes files that are Safe-Synthesizer-specific: `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `LICENSE`, `ruff.toml`, `pytest.ini`, `.pre-commit-config.yaml`, etc.

## Sync to NMP

Push Safe-Synthesizer changes back to NMP:

```bash
export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"

# Sync src/ and tests/ to NMP
make synchronize-py-files-to-nmp

# Sync metafiles too
make synchronize-to-nmp
```

After syncing to NMP, you'll need to commit and create an MR in the NMP repo separately.

## Post-Sync Workflow

After any sync operation, always run this sequence:

```bash
# 1. Review what changed
git status
git diff --stat

# 2. Format and check
make format
make check

# 3. Run tests
make test

# 4. Stage and commit (conventional commit format)
git add -A
git commit -s -m "chore: sync from NMP MR !<number>"

# 5. Push and create PR
git push -u origin HEAD
```

Then create a PR using the [github-cli skill](../github-cli/SKILL.md).

## Handling Non-Synced Files

The MR sync (`sync-from-mr.sh`) only mirrors `src/` and `tests/`. If the MR changed other files under `packages/nemo_safe_synthesizer/` (e.g., `design.md`, `script/`, docs), the script reports them. Handle manually:

```bash
# Check what was reported as non-synced, then copy manually
cp "$NMP_REPO_PATH/packages/nemo_safe_synthesizer/design.md" ./design.md
```

Or use the metafile sync target:

```bash
make synchronize-metafiles-from-nmp
```

## Finding NMP MRs to Sync

Use the GitLab MCP server to find merged MRs that touched the Safe-Synthesizer package. Alternatively, browse the NMP project merge requests at `gitlab-master.nvidia.com` filtering by path `packages/nemo_safe_synthesizer/`.

## Rsync Exclusions

Both directions exclude: `.git`, `.github`, `.vscode`, `.gitignore`, `.agent`, `__pycache__`, `*.pyc`, `.pytest_cache`, `.envrc`, `.venv`, `.cursor`.

Metafile sync additionally excludes Safe-Synthesizer-specific files: `__init__.py`, `.pre-commit-config.yaml`, `.markdownlint.json`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `DCO`, `Makefile`, `LICENSE`, `README.md`, `SECURITY.md`, `THIRD_PARTY.md`, `ruff.toml`, `pytest.ini`, `pyproject.toml`, `tools`, `uv.lock`.

## Common Issues

| Problem | Solution |
|---------|----------|
| `NMP_REPO_PATH is not set` | `export NMP_REPO_PATH="$HOME/dev/aire/microservices/nmp"` |
| `MR is not merged` | The MR must be merged in NMP before syncing with `sync-from-mr.sh` |
| `NMP repo not found` | Clone NMP or fix `NMP_REPO_PATH` |
| On `main` branch | `synchronize-from-nmp-mr` auto-creates a branch; other targets don't |
| Merge conflicts after sync | Resolve manually, then `make format && make check && make test` |

## Detailed Workflows

For step-by-step recipes covering end-to-end scenarios (e.g., syncing multiple MRs, verifying NMP MR before syncing, full release sync), see [workflows.md](references/workflows.md).

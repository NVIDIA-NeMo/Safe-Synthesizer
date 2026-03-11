<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .agents/

Agent-neutral skill definitions. Skills here are available to any agent (Cursor, Claude Code, etc.) and are not tied to a specific tool's config format.

## Directory layout

```
.agents/
└── skills/             # One subdirectory per skill, each containing SKILL.md
    ├── configurator/
    ├── diagnose-failures/
    ├── git-worktrees/
    ├── github-cli/
    ├── python-observability/
    ├── sync-agent-config/
    ├── sync-with-nmp/
    ├── usage/
    └── uv-build/
```

## Skills

Each skill is a self-contained directory with a `SKILL.md` that an agent reads on demand. Skills provide domain knowledge and step-by-step workflows for recurring tasks.

| Skill | Purpose |
|-------|---------|
| `configurator` | Pydantic-to-Click parameter mapping and configurator patterns |
| `diagnose-failures` | Systematic failure diagnosis using the error hierarchy |
| `git-worktrees` | Git worktree workflows, DCO/GPG signing, Cursor worktree automation |
| `github-cli` | `gh` CLI usage for PRs, issues, and CI |
| `python-observability` | Structured logging with `CategoryLogger` and `TracedContext` |
| `sync-agent-config` | Syncing agent config changes between repos |
| `sync-with-nmp` | Syncing with the NMP upstream repository |
| `usage` | Usage tracking and reporting patterns |
| `uv-build` | Building and publishing Python packages with `uv` |

## Discoverability

Cursor natively scans `.agents/skills/` as a first-class skill location -- no symlinks or duplication needed. Claude Code and other agents also read skills directly from this directory.

## Adding skills

`.agents/skills/` is listed in `.gitignore`, so new skill directories must be force-added:

```bash
git add -f .agents/skills/<skill-name>/
```

For machine-local or personal skills that should never be committed, use `.agents/skills/personal/` -- it is gitignored without needing a force-add, so anything placed there stays local automatically.

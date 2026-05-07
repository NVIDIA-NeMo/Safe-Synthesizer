<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .agents/

Agent-neutral skill definitions. Skills here are available to any agent (Cursor, Claude Code, etc.) and are not tied to a specific tool's config format.

## Directory layout

```text
.agents/
└── skills/             # One subdirectory per skill, each containing SKILL.md
    ├── git-worktrees/
    ├── github-cli/
    ├── safe-synthesizer/
    └── uv-build/
```

## Skills

Each skill is a self-contained directory with a `SKILL.md` that an agent reads on demand. Skills provide domain knowledge and step-by-step workflows for recurring tasks.

| Skill | Purpose |
| ----- | ------- |
| `git-worktrees` | Git worktree workflows, DCO/GPG signing, Cursor worktree automation |
| `github-cli` | `gh` CLI usage for PRs, issues, and CI |
| `safe-synthesizer` | Usage-facing router for CLI/SDK runs, config, troubleshooting, and artifacts |
| `uv-build` | Building and publishing Python packages with `uv` |

Detailed implementation guidance that used to live in skills now lives in
developer docs:

- [Configuration Management](../docs/developer-guide/configuration_management.md)
- [Observability](../docs/developer-guide/observability.md)

## Discoverability

Cursor natively scans `.agents/skills/` as a first-class skill location -- no symlinks or duplication needed. Claude Code and other agents also read skills directly from this directory.

## Adding skills

`.agents/skills/` is listed in `.gitignore`, so new skill directories must be force-added:

```bash
git add -f .agents/skills/<skill-name>/
```

For machine-local or personal skills that should never be committed, use `.agents/skills/personal/` -- it is gitignored without needing a force-add, so anything placed there stays local automatically.

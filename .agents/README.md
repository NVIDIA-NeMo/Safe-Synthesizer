<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .agents/

Agent-neutral skill definitions. Skills here are available to Cursor, Claude
Code, Codex, and other agents without depending on machine-installed skills or
a specific tool's configuration format.

## Directory layout

```text
.agents/
├── commands/           # Shared development command guides
├── hooks/              # Shared client hook implementations
└── skills/             # One discovery entry per skill
    ├── git-worktrees/  # Includes the shared worktree setup script
    ├── github-cli/
    ├── safe-synthesizer@ -> ../../skills/safe-synthesizer
    └── uv-build/
```

## Skills

Each skill resolves to a self-contained directory with a `SKILL.md` that an
agent reads on demand. Skills can bundle scripts for deterministic repository
operations.

| Skill | Purpose |
| ----- | ------- |
| `git-worktrees` | Standalone worktree workflow, cross-agent environment setup, and DCO/GPG signing |
| `github-cli` | `gh` CLI usage for PRs, issues, and CI |
| `safe-synthesizer` | Usage-facing router for CLI/SDK runs, config, troubleshooting, and artifacts |
| `uv-build` | Building and publishing Python packages with `uv` |

Detailed implementation guidance that used to live in skills now lives in
developer docs:

- [Configuration Management](../docs/developer-guide/configuration_management.md)
- [Observability](../docs/developer-guide/observability.md)

## Discoverability

Cursor natively scans `.agents/skills/` as a first-class skill location. Claude
Code, Codex, and other agents can also read skills directly from this directory.
Most skills live there directly. The publishable `safe-synthesizer` package
lives under `skills/` and uses a relative symlink from `.agents/skills/` for
repository-local discovery.

Client directories expose shared commands and hooks through relative symlinks.
Keep reusable implementations under `.agents/`; keep only client registration
and client-specific formats under `.cursor/` or `.claude/`.

## Adding skills

`.agents/skills/` is listed in `.gitignore`, so new skill directories must be force-added:

```bash
git add -f .agents/skills/<skill-name>/
```

For machine-local or personal skills that should never be committed, use `.agents/skills/personal/` -- it is gitignored without needing a force-add, so anything placed there stays local automatically.

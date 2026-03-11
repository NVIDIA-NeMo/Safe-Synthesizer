<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .claude/

Claude Code-specific agent configuration: hook registrations and slash commands.

## Directory layout

```
.claude/
├── settings.json   # Hook registrations for Claude Code (mirrors .cursor/hooks.json for Cursor)
└── commands/       # Slash commands -- invoked as /command-name in Claude Code
    ├── bootstrap.md
    ├── build-docs.md
    ├── build-wheel.md
    ├── format.md
    ├── gpu-test.md
    ├── lint.md
    ├── pre-commit.md
    ├── start-docs-server.md
    ├── sync-nmp.md
    ├── test-ci-container.md
    ├── test-slow.md
    └── unit-test.md
```

## settings.json

Registers hook scripts against Claude Code lifecycle events. The hook scripts themselves live in `.cursor/hooks/` and are shared with Cursor.

| Event | Script | Purpose |
|-------|--------|---------|
| `SessionStart` | `session_context.sh` | Reports venv state; runs `uv sync --frozen` if `.venv` absent |
| `SessionEnd` | `audit.sh` | Appends audit entry |
| `PreToolUse` (Bash) | `enforce-signoff.sh` | Blocks commits missing `--signoff` or `--gpg-sign` |
| `PreToolUse`, `PostToolUse`, `PreCompact`, `Stop` | `audit.sh` | Audit log |

## commands/

Each file is a slash command available in Claude Code as `/command-name` (filename without `.md`). Commands map common development tasks to the correct Make targets and tool invocations for this repo. The `claude-commands.mdc` rule in `.cursor/rules/` also surfaces these to Cursor agents by keyword.

| Command | Task |
|---------|------|
| `bootstrap` | Bootstrap dev environment |
| `build-docs` | Build documentation site |
| `build-wheel` | Build Python wheel |
| `format` | Format code (ruff + copyright) |
| `gpu-test` | GPU integration and e2e tests |
| `lint` | Lint and typecheck |
| `pre-commit` | Run all pre-commit hooks |
| `start-docs-server` | Local docs dev server |
| `sync-nmp` | Sync with NMP repository |
| `test-ci-container` | CI tests in a container |
| `test-slow` | All tests including slow |
| `unit-test` | Run unit tests |

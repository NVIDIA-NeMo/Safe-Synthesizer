<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .claude/

Claude Code-specific agent configuration: hook registrations and slash commands.

## Directory layout

```
.claude/
├── settings.json   # Hook registrations for Claude Code (mirrors .cursor/hooks.json for Cursor)
├── hooks/
│   └── enforce-signoff.sh@  # Symlink to the shared signing hook
└── commands@       # Symlink to ../.agents/commands; invoked as /command-name
    ├── bootstrap.md
    ├── build-docs.md
    ├── build-wheel.md
    ├── format.md
    ├── gpu-test.md
    ├── lint.md
    ├── start-docs-server.md
    ├── test-ci-container.md
    ├── test-slow.md
    └── unit-test.md
```

## settings.json

Registers the shared signing hook against Claude Code lifecycle events.

| Event | Script | Purpose |
|-------|--------|---------|
| `PreToolUse` (Bash) | `enforce-signoff.sh` | Blocks commits missing `--signoff` or `--gpg-sign` |

## commands/

Each file is a slash command available in Claude Code as `/command-name`
(filename without `.md`). Commands map common development tasks to the correct
mise tasks and tool invocations for this repo. The `agent-commands.mdc` rule in
`.cursor/rules/` also surfaces them to Cursor agents by keyword.

| Command | Task |
|---------|------|
| `bootstrap` | Bootstrap dev environment |
| `build-docs` | Build documentation site |
| `build-wheel` | Build Python wheel |
| `format` | Format code (ruff + copyright) |
| `gpu-test` | GPU integration and e2e tests |
| `lint` | Lint and typecheck |
| `start-docs-server` | Local docs dev server |
| `test-ci-container` | CI tests in a container |
| `test-slow` | All unit tests including slow |
| `unit-test` | Run unit tests |

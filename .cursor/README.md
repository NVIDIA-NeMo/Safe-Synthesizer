<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .cursor/

Cursor-specific agent configuration: hooks, rules, skills, and worktree setup.

## Directory layout

```
.cursor/
├── hooks.json          # Hook registrations for Cursor (mirrors .claude/settings.json for Claude Code)
├── worktrees.json      # Parallel-agent worktree setup script pointer
├── setup-worktree.sh   # Runs inside each new parallel-agent worktree at creation time
├── hooks/              # Hook scripts (shared by both Cursor and Claude Code)
│   ├── session_context.sh   # sessionStart -- reports venv state, runs uv sync if .venv absent
│   ├── enforce-signoff.sh   # beforeShellExecution/PreToolUse(Bash) -- blocks git commit without --signoff/-s and --gpg-sign/-S
│   └── audit.sh             # most events -- appends timestamped JSON to ~/.cursor/audit.log
├── rules/              # Always-apply and requestable context rules (.mdc files)
│   ├── agent-markdown-style.mdc   # alwaysApply -- markdown and docstring style conventions
│   ├── claude-commands.mdc        # alwaysApply -- maps task keywords to .claude/commands/ files
│   ├── repo-navigation.mdc        # requestable -- repo layout, skills, tests, config files
│   └── writing-docs.mdc           # requestable -- documentation writing conventions
├── agents/             # Named subagent persona definitions
│   └── verifier.md     # Skeptical verification agent (fast model, run after task completion)
└── skills/             # Empty -- Cursor discovers skills directly from .agents/skills/
```

## hooks.json

Registers hook scripts against Cursor lifecycle events.

| Event | Script | Purpose |
|-------|--------|---------|
| `sessionStart` | `session_context.sh` | Reports venv state; runs `uv sync --frozen` if `.venv` absent |
| `sessionEnd` | `audit.sh` | Appends audit entry |
| `beforeShellExecution` (git commit) | `enforce-signoff.sh` | Blocks commits missing `--signoff` or `--gpg-sign` |
| `beforeShellExecution`, `beforeMCPExecution`, `afterShellExecution`, `afterMCPExecution`, `afterFileEdit`, `beforeSubmitPrompt`, `preCompact`, `stop` | `audit.sh` | Audit log |

The same hook scripts are registered in `.claude/settings.json` for Claude Code using its `PreToolUse`/`PostToolUse` event model.

## worktrees.json

Points Cursor to `setup-worktree.sh`, which runs inside each new parallel-agent worktree at creation time. Standard JSON -- no JSONC comments supported.

## skills/

Empty directory. Cursor natively scans both `.cursor/skills/` and `.agents/skills/` as first-class skill locations, so no symlinks are needed.

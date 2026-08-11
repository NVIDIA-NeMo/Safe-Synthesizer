<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# .cursor/

Cursor-specific agent configuration: hooks, rules, skills, and worktree setup.

## Directory layout

```
.cursor/
├── hooks.json          # Hook registrations for Cursor (mirrors .claude/settings.json for Claude Code)
├── worktrees.json      # Points to the agent-neutral worktree setup script
├── commands@           # Symlink to ../.agents/commands
├── hooks/
│   └── enforce-signoff.sh@  # Symlink to the shared signing hook
├── rules/              # Always-apply and requestable context rules (.mdc files)
│   ├── agent-markdown-style.mdc   # alwaysApply -- markdown and docstring style conventions
│   ├── agent-commands.mdc         # alwaysApply -- maps task keywords to .agents/commands/ files
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
| `beforeShellExecution` | `enforce-signoff.sh` | Blocks recognized Git commits missing `--signoff` or `--gpg-sign` |

Cursor invokes the hook for every shell command so it can recognize direct Git
commits after environment assignments, command separators, and Git global
options. Nested interpreters and other process wrappers are outside this
best-effort guardrail's scope. The hook exits immediately for other commands
and requires the repository-pinned `jq` tool. Trust the workspace before
relying on project hooks, and verify that `.cursor/hooks.json` is active.

Use staged commits with standalone signing flags, such as
`git commit -s -S -m "message"`. The guard rejects non-message quoted arguments
and explicit signing negations rather than attempting to interpret them.

The same hook implementation is exposed to Claude Code through a client-local
symlink and registered directly for Codex in `.codex/hooks.json`.

## worktrees.json

Points Cursor to the setup script bundled with the `git-worktrees` skill. The
script creates a local virtual environment but does not copy `.env`,
`.env.local`, `mise.local.toml`, or other machine-local configuration from
another checkout. Standard JSON only; JSONC comments are not supported.

## skills/

Empty directory. Cursor natively scans both `.cursor/skills/` and `.agents/skills/` as first-class skill locations, so no symlinks are needed.

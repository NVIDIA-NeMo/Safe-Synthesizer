---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: git-worktrees
description: "Create, clean up, or prune isolated Safe-Synthesizer git worktrees for feature development, PR review, and parallel branches across Cursor, Claude Code, Codex, and plain shells."
license: Apache-2.0
---

# Git Worktrees

This skill is self-contained. Do not load a machine-installed or system
worktree skill. Apply standard Git behavior except for the repository rules
below.

## Repository rules

- Preserve unrelated changes. Do not stash, move, or discard them to create a
  worktree.
- Do not depend on client hooks. Cursor invokes
  `scripts/setup-worktree.sh` through `.cursor/worktrees.json`. Other clients
  can run the same script after creating a worktree. Inspect the actual
  environment and machine-local configuration without printing secrets.
- Codex does not read `.cursor/worktrees.json`; configure its local environment
  to invoke `scripts/setup-worktree.sh`, or run the script manually.
- Keep machine-local configuration local to each checkout. Worktree hooks do
  not copy `.env`, `.env.local`, `mise.local.toml`, or `.local.envrc` from
  another checkout.

## Python environment

Run the setup script from a new worktree if the client did not run it:

```bash
.agents/skills/git-worktrees/scripts/setup-worktree.sh
```

It creates a base local `.venv`. Select the complete dependency profile needed
for development:

```bash
unset UV_PROJECT_ENVIRONMENT UV_NO_SYNC PYTHONPATH VIRTUAL_ENV UV_PROJECT_DIR
mise run setup
mise run bootstrap-nss cpu  # or cu129
```

Bare `uv sync --frozen` installs only the base environment. It does not support
all type checks, import checks, or GPU tests.

Share the main checkout's environment only when the lock and dependency
profile match:

```bash
SS_MAIN_CHECKOUT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
export UV_PROJECT_ENVIRONMENT="$SS_MAIN_CHECKOUT/.venv"
export UV_NO_SYNC=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The main checkout owns synchronization of a shared environment. Do not run
`uv sync` or a bootstrap task from a worktree that points to it.

## Commits

Commits require DCO sign-off and GPG signing:

```bash
git commit --signoff --gpg-sign -m "message"
```

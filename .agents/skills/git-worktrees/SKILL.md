# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
name: git-worktrees
description: "Create, manage, and clean up git worktrees for isolated development, PR review, and A/B testing of agent configurations. Trigger keywords - worktree, worktrees, git worktree, parallel branches, isolated workspace, worktree cleanup, worktree prune, PR review, address PR comments, work on branch, work on PR."
license: Apache-2.0
---

# Git Worktrees (Safe-Synthesizer)

Safe-Synthesizer-specific additions to the base `git-worktrees` skill. Read the base skill first (available globally via agent-stuff), then apply the overrides below.

## Virtual Environments

### Cursor parallel-agent worktrees (automatic)

Cursor parallel-agent worktrees created via the IDE run `.cursor/setup-worktree.sh` automatically at creation time. This script runs `uv sync --frozen` unconditionally, producing a local `.venv` in the worktree. No manual setup needed.

The `sessionStart` hook (`.cursor/hooks/session_context.sh`) also runs `uv sync --frozen` if `.venv` is absent at session start, covering Claude Code worktree sessions.

### Manual worktrees (agent-created)

When creating a worktree manually (e.g., for PR review or feature work), always run `uv sync --frozen` after creation:

```bash
cd "$SS_WORKTREE_DIR/ss-wt-<name>"
uv sync --frozen
```

This creates a local `.venv` in the worktree. With uv's cache the install takes ~2-3 seconds on a warm cache.

If you need different extras (e.g. `cu128` vs `cpu`), pass them explicitly:

```bash
uv sync --frozen --extra cu128 --extra engine --group dev
```

Never run bare `uv sync` without `--frozen` -- it re-locks `uv.lock` and creates dirty state.

Note: The base skill describes sharing the main repo's `.venv` via `UV_PROJECT_ENVIRONMENT` as the default. In this repo, the Cursor worktree automation always creates a local `.venv`, so `UV_PROJECT_ENVIRONMENT` is not needed for Cursor-managed worktrees. Use it only for quick throwaway sessions where you know the deps are identical.

## Commit Requirements

Every commit requires both DCO sign-off and GPG signing. The `enforce-signoff.sh` hook blocks commits that omit either:

```bash
git commit --signoff --gpg-sign -m "message"
# or short flags:
git commit -s -S -m "message"
```

Never hand-write the `Signed-off-by` trailer -- it must come from `--signoff` so it matches `git config user.name` / `user.email` exactly.

GPG signing requires a key configured in git:

```bash
# Check current signing key
git config --global user.signingkey

# Set a key (replace with your key ID or SSH key path)
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgSign true

# For SSH signing (simpler than GPG for new setups)
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgSign true
```

GitHub marks commits as "Verified" when the signing key matches a key registered in your GitHub account (Settings → SSH and GPG keys). Commits squash-merged by GitHub are signed by GitHub's own key -- only locally-created commits need your personal key.

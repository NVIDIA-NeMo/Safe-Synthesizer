---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: git-worktrees
description: "Create, manage, and clean up git worktrees for isolated development, PR review, and A/B testing of agent configurations. Trigger keywords - worktree, worktrees, git worktree, parallel branches, isolated workspace, worktree cleanup, worktree prune."
---

# Git Worktrees

Git worktrees create isolated working directories that share the same `.git` repository. Each worktree has its own checked-out branch, index, and working tree -- but commits, refs, and objects are shared. This avoids the overhead of a full clone while letting you work on multiple branches simultaneously.

## Shell Permissions

Always use `required_permissions: ["all"]` when running `git worktree add`. It writes outside the workspace directory and will fail in a sandbox.

## Worktree Base Directory

Worktrees are created as siblings of the repo checkout. The base directory defaults to the parent of the repo root, overridable via `SS_WORKTREE_DIR`:

```bash
SS_WORKTREE_DIR="${SS_WORKTREE_DIR:-$(dirname "$(git rev-parse --show-toplevel)")}"
```

Set `SS_WORKTREE_DIR` in your shell profile or `AGENTS.local.md` if your layout differs.

Note: `git rev-parse --show-toplevel` resolves symlinks. If the repo is accessed via a symlink, the default base directory will be relative to the real path, not the symlink. Set `SS_WORKTREE_DIR` explicitly if that matters.

## Virtual Environments

By default, worktrees share the main repo's `.venv` via `UV_PROJECT_ENVIRONMENT`. This is instant -- no install step needed.

```bash
MAIN_WORKTREE="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
export UV_PROJECT_ENVIRONMENT="$MAIN_WORKTREE/.venv"
```

`UV_PROJECT_ENVIRONMENT` tells `uv run` and `uv sync` to use the specified venv instead of creating a local `.venv`. With this set, `uv run pytest`, `uv run python`, etc. all use the main repo's packages.

If the worktree changes `pyproject.toml` or needs different extras, create its own venv instead:

```bash
unset UV_PROJECT_ENVIRONMENT
uv sync --frozen
```

The `--frozen` flag installs from the existing `uv.lock` without re-locking. Never run bare `uv sync` in a worktree -- it can silently modify the tracked `uv.lock`, creating dirty state.

## Naming Conventions

All worktrees use the `ss-wt-` prefix (`ss` = Safe-Synthesizer, `wt` = worktree):

| Pattern | Purpose | Example |
|---------|---------|---------|
| `$SS_WORKTREE_DIR/ss-wt-<name>` | Feature branch work, PR review | `ss-wt-gh-skill`, `ss-wt-signoff-hooks` |
| `${TMPDIR:-/tmp}/ss-wt-<name>` | Ephemeral / throwaway | `ss-wt-quick-test` |

The `ss-wt-` prefix makes worktree directories immediately identifiable as belonging to this repo, even when listed alongside unrelated directories.

## Creation Workflow

### Step 1: Check for Existing Worktrees

```bash
git worktree list
```

Reuse an existing worktree if one already tracks the branch you need. To repurpose a stale worktree for new work:

```bash
cd "$SS_WORKTREE_DIR/ss-wt-<name>"
git checkout main && git pull
git checkout -b <user>/<issue>-<new-name> origin/main
```

### Step 2: Create the Worktree

```bash
SS_WORKTREE_DIR="${SS_WORKTREE_DIR:-$(dirname "$(git rev-parse --show-toplevel)")}"

# New feature branch off main
git worktree add "$SS_WORKTREE_DIR/ss-wt-<name>" -b <user>/<issue>-<name> origin/main

# Existing branch
git worktree add "$SS_WORKTREE_DIR/ss-wt-<name>" <branch>

# Ephemeral
git worktree add "${TMPDIR:-/tmp}/ss-wt-<name>" <branch>
```

### Step 3: Set Up the Environment

By default, reuse the main repo's venv (no install needed):

```bash
MAIN_WORKTREE="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
export UV_PROJECT_ENVIRONMENT="$MAIN_WORKTREE/.venv"
```

If the worktree has different dependencies (modified `pyproject.toml`, different extras), create its own venv instead:

```bash
unset UV_PROJECT_ENVIRONMENT
uv sync --frozen
```

Never run bare `uv sync` without `--frozen` -- it can silently re-lock `uv.lock`.

### Step 4: Verify Baseline

```bash
make test
```

If tests fail, investigate before proceeding -- you need a clean baseline to distinguish pre-existing failures from your changes.

### Step 5: Confirm

Report the worktree location and branch:

```bash
git worktree list --porcelain | head -6
```

## Safety Checks

Before creating a worktree:

1. Target path must not already exist (`ls "$SS_WORKTREE_DIR/ss-wt-<name>"` should fail)
2. Branch name must not already be checked out in another worktree (git enforces this)
3. Main worktree should have clean status (`git status --short` is empty) -- avoids confusion about where uncommitted changes live
4. Run from the main repo checkout, not from inside another worktree -- `git rev-parse --show-toplevel` returns the current worktree's root, which would put the new worktree in the wrong place

For project-local worktree directories (not currently used in this repo, but if you ever place worktrees inside the repo tree):

```bash
git check-ignore -q .worktrees 2>/dev/null || echo "WARNING: add .worktrees to .gitignore"
```

## Cleanup and Pruning

Worktrees accumulate over time. Periodic cleanup prevents confusion and reclaims disk space.

### Audit Active Worktrees

```bash
git worktree list
```

Each line shows `<path> <commit> [<branch>]`. Look for:

- Branches that have been merged to `main`
- Worktrees for experiments that finished
- Paths pointing to directories that no longer exist (stale entries)
- Locked worktrees (shows `locked` in the output)

### Remove a Worktree

```bash
# Safe removal -- refuses if there are modified tracked files or staged changes
git worktree remove "$SS_WORKTREE_DIR/ss-wt-<name>"

# Force removal -- discards all changes including untracked files
git worktree remove --force "$SS_WORKTREE_DIR/ss-wt-<name>"
```

This deletes the directory and removes the entry from `.git/worktrees/`.

If the worktree is locked, unlock it first:

```bash
git worktree unlock "$SS_WORKTREE_DIR/ss-wt-<name>"
git worktree remove "$SS_WORKTREE_DIR/ss-wt-<name>"
```

### Delete the Branch (if merged)

After removing the worktree, the branch still exists:

```bash
# Only deletes if fully merged to current branch
git branch -d <branch-name>

# Force-delete regardless of merge status
git branch -D <branch-name>
```

### Prune Stale Entries

If a worktree directory was deleted manually (e.g., `rm -rf "$SS_WORKTREE_DIR/ss-wt-<name>"`), git still has a stale entry in `.git/worktrees/`. Clean it up:

```bash
git worktree prune
```

### Full Cleanup Recipe

```bash
# 1. List everything
git worktree list

# 2. Remove worktrees you no longer need
git worktree remove "$SS_WORKTREE_DIR/ss-wt-<name>"

# 3. Prune any orphaned entries
git worktree prune

# 4. Delete merged branches
merged=$(git branch --merged main | grep -E -v '^\*|main')
[ -n "$merged" ] && echo "$merged" | xargs git branch -d
```

## Agent Gotchas

Lessons from real sessions working in this repo:

- Sandbox permissions: `git worktree add` writes outside the workspace directory. Always use `required_permissions: ["all"]` in the Shell tool. Without it, the command fails silently or with a cryptic permission error.

- DCO sign-off: worktrees share the same `.git/config`, so `user.name` and `user.email` are inherited. Verify they match your expected identity before committing. Always use `git commit --signoff` (never hand-write the trailer).

- Shared stashes: stashes live in the shared `.git` directory, so they're visible from all worktrees. `git stash drop` in any worktree affects all of them. This is usually fine -- just be aware of it.

- Absolute paths: when editing files in a worktree from a different working directory (e.g., main workspace), always use absolute paths. Relative paths resolve against the current directory, not the worktree.

- Subagent context: subagents inherit the IDE workspace's rules and skills, not those of a target worktree. When dispatching subagent work into a different worktree, copy the relevant context (file paths, constraints, expected config) into the subagent prompt verbatim.

- One branch per worktree: git does not allow the same branch to be checked out in multiple worktrees simultaneously. If you need to work on the same branch from two places, use `git worktree add --detach` for a detached HEAD, or finish work in the first worktree first.

- Worktree-from-worktree: `git rev-parse --show-toplevel` returns the current worktree's root, not the main repo root. Always create new worktrees from the main checkout to get correct `SS_WORKTREE_DIR` resolution.

- Dependency safety: worktrees default to sharing the main repo's `.venv` via `UV_PROJECT_ENVIRONMENT`. If the worktree changes `pyproject.toml`, unset it and run `uv sync --frozen` to create a local venv. Never run bare `uv sync` in a worktree -- it can silently modify the tracked `uv.lock` file.

## Worktree to Draft PR

Common pattern: make changes in a worktree and open a draft PR in one chain.

```bash
cd "$SS_WORKTREE_DIR/ss-wt-<name>" \
  && git add -A && git commit -s -m "feat: description" \
  && git push -u origin HEAD \
  && gh pr create --draft --title "feat: description" --body "$(cat <<'EOF'
## Summary
- ...
EOF
)"
```

When dispatching this workflow, ask the user whether they want a worktree or a local branch. Worktrees are preferred when:

- The main checkout has uncommitted work or is on a different branch
- Multiple branches need to be active simultaneously
- The user explicitly requested a worktree

## Quick Reference

| Situation | Command |
|-----------|---------|
| Set base dir | `SS_WORKTREE_DIR="${SS_WORKTREE_DIR:-$(dirname "$(git rev-parse --show-toplevel)")}"` |
| List worktrees | `git worktree list` |
| New feature branch | `git worktree add "$SS_WORKTREE_DIR/ss-wt-<name>" -b <branch> origin/main` |
| Existing branch | `git worktree add "$SS_WORKTREE_DIR/ss-wt-<name>" <branch>` |
| Throwaway | `git worktree add "${TMPDIR:-/tmp}/ss-wt-<name>" <branch>` |
| Share main venv | `export UV_PROJECT_ENVIRONMENT="$MAIN_WORKTREE/.venv"` |
| Isolated venv | `uv sync --frozen` (when deps differ) |
| Verify baseline | `cd <path> && make test` |
| Done with worktree | `git worktree remove <path>` |
| Dir deleted manually | `git worktree prune` |
| Branch merged | `git worktree remove <path> && git branch -d <branch>` |

## Common Mistakes

- Don't create a worktree without checking `git worktree list` first -- reuse existing ones
- Don't forget `required_permissions: ["all"]` -- worktree creation writes outside the sandbox
- Don't skip baseline verification (`make test`) -- you need a clean starting point
- Don't manually delete worktree directories with `rm -rf` -- use `git worktree remove` so git cleans up `.git/worktrees/` entries too
- Don't create worktrees from inside another worktree -- `--show-toplevel` resolves to the wrong root
- Don't run bare `uv sync` in a worktree -- use `uv sync --frozen` if you need a local venv, or share the main venv via `UV_PROJECT_ENVIRONMENT`
- Always use the `ss-wt-` prefix for worktree directory names

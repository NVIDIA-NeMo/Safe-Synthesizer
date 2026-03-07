---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: github-cli
description: "Interact with the Safe-Synthesizer GitHub repository using the gh CLI. Activate when users want to list or create pull requests, check out PRs, work on someone else's PR, check CI status, investigate workflow failures, view job logs, create or triage issues, check review and approval status, manage releases, or inspect repo metadata. Trigger keywords - pull request, PR, issue, workflow, CI, actions, failed job, job log, release, review, approve, CODEOWNERS, labels, milestone, checkout, gh, GitHub."
---

# GitHub CLI (gh)

## Detailed References

- [Full Reference](./references/full-reference.md) - Comprehensive `gh` reference for PRs, issues, Actions/CI, code review, releases, and repo metadata
- [Workflows](./references/workflows.md) - Pre-merge checklist, debug failed CI, release workflow, triage issues, review PRs locally, fetch and reply to PR comments

Note: The commands below are quick references. For comprehensive detail and multi-step workflows, see the references above.

## Scripts

**PR comments: use the CLI helper** (PEP 723 + Typer + PyGithub + Pydantic). Run from repo root or from this skill directory. No `gh` binary required for fetch/reply (uses GitHub API with `GITHUB_TOKEN` or `--token`; optional fallback: `gh auth token`). If `GITHUB_TOKEN` is not set, run `export GITHUB_TOKEN=$(gh auth token)` before invoking the helper (or pass `--token`). Requires network; in Agent use `required_permissions: ["all"]` per [sandbox behavior](https://cursor.com/docs/agent/tools/terminal).

Path from repo root: `.agent/skills/github-cli/scripts/gh_pr_helper.py` (or `scripts/gh_pr_helper.py` if symlinked).

| Command | Description |
|---------|-------------|
| `uv run --script .agent/skills/github-cli/scripts/gh_pr_helper.py -- comments [PR_NUMBER]` | Fetch all PR comments (inline + top-level). Single JSON object to stdout: `{ "pr_number", "repo", "inline": [...], "top_level": [...] }`. Omit PR to use current branch’s open PR. |
| `uv run --script .agent/skills/github-cli/scripts/gh_pr_helper.py -- reply <COMMENT_ID> "Reply body"` | Post a reply to an inline review comment. |
| `uv run --script ... -- reply <COMMENT_ID> --reply-file -` | Reply body from stdin. |
| `uv run --script ... -- reply <COMMENT_ID> --reply-file path/to/body.md` | Reply body from file. |

Draft the reply with the user in a file, then run with `--reply-file path` to post (avoids long inline strings).

Options (all commands): `--repo OWNER/REPO` (default: from git remote), `--token` / `-t` (default: `GITHUB_TOKEN` or `gh auth token`). Full workflow: [references/workflows.md](./references/workflows.md) § Fetch and Address Review Comments.

## Shell Permissions

Always use `required_permissions: ["all"]` when running `gh` commands. Sandboxed environments fail with TLS certificate errors (`x509: OSStatus -26276`).

## Prefer && Chains

When running multiple sequential commands (pre-flight, commit, push, PR create), chain them with `&&` in a single shell call. Only use separate calls when you need to read intermediate output before deciding the next step.

## Pre-flight

Before any `gh` operation, verify the binary is available and authenticated:

```bash
export PATH="$HOME/.local/bin:$PATH" && gh auth status
```

## Pull Requests

```bash
# List open PRs
gh pr list
gh pr list --author=@me

# View PR details
gh pr view <number>
gh pr view <number> --json title,state,reviews,statusCheckRollup

# Create PR
gh pr create --title "Title" --body "Description"

# Check out a PR locally
gh pr checkout <number>

# Create a draft PR (WIP)
gh pr create --draft --title "Title" --body "Description"

# View PR diff
gh pr diff <number>

# Check if a PR already exists for the current branch before creating
gh pr view --json number,url 2>/dev/null && echo "PR exists" || echo "No PR yet"

# Edit an existing PR's title/body
gh pr edit <number> --title "New title" --body "New body"

# Commit, push, and create PR in one call
git add -A && git commit -s -S -m "feat: short title" \
  && git push -u origin HEAD \
  && gh pr create --draft --title "feat: short title" --body "$(cat <<'EOF'
## Summary
- ...
EOF
)"
```

## CI / Actions

```bash
# Check CI status + recent runs in one call
gh pr checks && gh run list --limit 5

# When you already have the run ID (e.g. from a URL the user pasted):
gh run view <run-id> --log-failed

# Summary + logs in one call
gh run view <run-id> && gh run view <run-id> --log-failed

# Find latest failure on current branch (when no run ID given)
RUN_ID=$(gh run list --branch="$(git branch --show-current)" --status=failure --limit=1 --json databaseId -q '.[0].databaseId') \
  && [ "$RUN_ID" != "null" ] && gh run view "$RUN_ID" --log-failed
```

## Issues

```bash
gh issue list
gh issue view <number>
gh issue create --title "Title" --body "Description"

# Edit an existing issue
gh issue edit <number> --body "Updated body"
```

## Code Review

```bash
# List reviews on a PR
gh pr view <number> --json reviews

# Approve
gh pr review <number> --approve

# Request changes
gh pr review <number> --request-changes --body "feedback"
```

## Releases

```bash
gh release list
gh release view <tag>
gh release create <tag> --generate-notes
```

## Writing PR and Issue Bodies

Always use HEREDOC for multiline bodies:

```bash
gh pr create --title "feat: short title" --body "$(cat <<'EOF'
## Summary
- 2-4 bullet points, not a full audit

## Test plan
- [ ] Verify X
- [ ] Check Y
EOF
)"
```

Keep bodies concise -- 2-4 bullet summary for PRs, problem + options for issues. Don't generate long audit dumps or parameter inventories. When the user asks for "succinct" -- 1-3 sentences, no lists.

## Common Mistakes

- Don't run `gh run view <id>` and `gh run view <id> --log-failed` as separate calls -- go straight to `--log-failed` when you already have the run ID
- Don't WebFetch GitHub Actions URLs for private repos -- use `gh run view` instead (auth required)
- Don't propose `gh pr create` without checking if a PR already exists first
- Don't generate long PR/issue bodies -- users consistently ask agents to cut them down
- Don't use separate shell calls for `git push` then `gh pr create` -- chain them with `&&`
- Don't manually write `Signed-off-by` in commit messages -- always use `git commit --signoff --gpg-sign` (`-s -S`) so the trailer matches `git config user.name` / `user.email` and the commit is cryptographically signed; DCO probot and signature verification both require exact identity match

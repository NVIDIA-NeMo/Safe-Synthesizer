# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
name: github-cli
description: "Interact with the Safe-Synthesizer GitHub repository using the gh CLI. Activate when users want to list or create pull requests, check out PRs, work on someone else's PR, check CI status, investigate workflow failures, view job logs, create or triage issues, check review and approval status, manage releases, or inspect repo metadata. Trigger keywords - pull request, PR, issue, workflow, CI, actions, failed job, job log, release, review, approve, CODEOWNERS, labels, milestone, checkout, gh, GitHub."
license: Apache-2.0
---

# GitHub CLI (gh)

Multi-step workflows (pre-merge checklist, debug CI, release, fetch and reply to PR comments): [references/workflows.md](./references/workflows.md).

## Scripts

**PR comments: use the CLI helper** (PEP 723 + Typer + PyGithub + Pydantic). Run from repo root or from this skill directory. No `gh` binary required for fetch/reply (uses GitHub API with `GITHUB_TOKEN` or `--token`; optional fallback: `gh auth token`). If `GITHUB_TOKEN` is not set, run `export GITHUB_TOKEN=$(gh auth token)` before invoking the helper (or pass `--token`). Requires network; in Agent use `required_permissions: ["all"]` per [sandbox behavior](https://cursor.com/docs/agent/tools/terminal).

Path from repo root: `.agents/skills/github-cli/scripts/gh_pr_helper.py` (or `scripts/gh_pr_helper.py` if symlinked).

| Command | Description |
|---------|-------------|
| `uv run --script .agents/skills/github-cli/scripts/gh_pr_helper.py -- comments [PR_NUMBER]` | Fetch all PR comments (inline + top-level). Single JSON object to stdout: `{ "pr_number", "repo", "inline": [...], "top_level": [...] }`. Omit PR to use current branch's open PR. |
| `uv run --script .agents/skills/github-cli/scripts/gh_pr_helper.py -- reply <COMMENT_ID> "Reply body"` | Post a reply to an inline review comment. |
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

# View PR details -- use --json to avoid GraphQL errors with deprecated fields
gh pr view <number> --json number,title,state,url,headRefName,baseRefName,isDraft,mergeable,reviewDecision,statusCheckRollup,additions,deletions,changedFiles,reviews,labels,assignees,author \
  --jq '{number, title, state, url, branch: .headRefName, base: .baseRefName, draft: .isDraft, mergeable, reviewDecision, author: .author.login, additions, deletions, changedFiles, labels: [.labels[].name], checks: [.statusCheckRollup[] | {name: .name, status: .status, conclusion: .conclusion}], reviews: [.reviews[] | {author: .author.login, state: .state}]}'

# Create PR
gh pr create --title "Title" --body "Description"

# Check out a PR locally
gh pr checkout <number>

# Create a draft PR (WIP)
gh pr create --draft --title "Title" --body "Description"

# View PR diff
gh pr diff <number>

# Check if a PR already exists before creating -- use gh pr edit if it does
gh pr view --json number,url 2>/dev/null && echo "PR exists -- use gh pr edit" || echo "No PR yet"

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

## Review Comments

`gh pr view --json comments` returns top-level PR conversation comments only. Inline code review comments require the REST API:

```bash
# Inline code review comments on a PR
gh api repos/NVIDIA-NeMo/Safe-Synthesizer/pulls/<number>/comments

# Issue discussion thread
gh api repos/NVIDIA-NeMo/Safe-Synthesizer/issues/<number>/comments

# Get a PR's base commit SHA (useful for lockfile diff diagnosis)
gh pr view <number> --json baseRefOid -q .baseRefOid
```

## CI / Actions

The project has 7 GitHub Actions workflows:

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| CI Checks | `ci-checks.yml` | push to main, PRs, manual | format, lint, typecheck, unit-test |
| GPU jobs | `gpu-tests.yml` | push to main, push to `pull-request/<N>`, manual | e2e on A100 (copy-pr-bot; see below) |
| Conventional Commit | `conventional-commit.yml` | PR title changes | Validates PR title format |
| Copyright Check | `copyright-check.yml` | push to main, release branches | NVIDIA copyright headers |
| DCO Assistant | `dco-assistant.yml` | PR events, comments | Developer Certificate of Origin |
| Release | `release.yml` | manual dispatch only | Build and publish to PyPI |
| Secrets Detector | `secrets-detector.yml` | PRs to main | Scans for leaked secrets |

GPU tests use the copy-pr-bot pattern: they do NOT fire on `pull_request` events. Instead, copy-pr-bot pushes PR content to a `pull-request/<N>` branch, which triggers the workflow via a push event. To manually trigger GPU tests on a PR, comment `/sync` on the PR. See the [copy-pr-bot docs](https://docs.gha-runners.nvidia.com/platform/apps/copy-pr-bot/).

```bash
# Check CI status + recent runs in one call
gh pr checks && gh run list --limit 5

# When you have a run ID (e.g. from a URL like .../actions/runs/<run-id>/job/<job-id>):
gh run view <run-id> --log-failed

# Find latest failure on current branch (when no run ID given)
RUN_ID=$(gh run list --branch="$(git branch --show-current)" --status=failure --limit=1 --json databaseId -q '.[0].databaseId') \
  && [ "$RUN_ID" != "null" ] && gh run view "$RUN_ID" --log-failed

# Re-run failed jobs only
gh run rerun <run-id> --failed
```

CI jobs map to local commands:

| CI Job | Local Command |
|--------|---------------|
| Format | `make format` (fix) or `make format-check` (check) |
| Format (lock) | `make lock-check` |
| Typecheck | `make typecheck` |
| Unit Tests | `make test-ci` or `make test` |

Path filtering may skip format/typecheck/unit-test when only non-source files change. Run `make check` and `make test` locally or add a trivial Python change to trigger CI.

## Issues

```bash
gh issue list
gh issue view <number>
gh issue create --title "Title" --body "Description"
gh issue edit <number> --body "Updated body"
```

Issue templates: `bug-report`, `feature-request`, `development-task` (use `gh issue create` for interactive prompt).

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

# Trigger release workflow (dry run first)
gh workflow run release.yml \
  -f release-ref=<full-sha-or-tag> \
  -f dry-run=true \
  -f create-gh-release=true \
  -f version-bump-branch=<branch>
```

## CODEOWNERS

Defined in `.github/CODEOWNERS`:

- All files: `@NVIDIA-NeMo/safe-synthesizer-maintainers`
- `src/` and `tests/`: `@NVIDIA-NeMo/safe-synthesizer-reviewers`
- Critical files (`pyproject.toml`, `uv.lock`, `SECURITY.md`, `LICENSE`, `.github/`): `@NVIDIA-NeMo/safe-synthesizer-maintainers`

```bash
gh pr edit <number> --add-reviewer NVIDIA-NeMo/safe-synthesizer-reviewers
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

Keep bodies concise -- 2-4 bullet summary for PRs, problem + options for issues. Don't generate long audit dumps or parameter inventories. When the user asks for "succinct" -- 1-3 sentences, no lists. For issues meant for team discussion, keep options human-readable and decision-focused, not implementation inventories.

No decorative bold (`**text**`) in PR or issue bodies. No per-file change inventories. No `## Why` sections that restate the commit message. Default to the shortest body that answers "what changed and how to verify it."

## Common Mistakes

- Don't run `gh run view <id>` and `gh run view <id> --log-failed` as separate calls -- go straight to `--log-failed` when you already have the run ID
- Don't WebFetch GitHub Actions URLs for private repos -- returns 404; use `gh run view <run-id> --log-failed` instead
- Don't propose `gh pr create` without checking if a PR already exists first -- use `gh pr edit` if it does
- Don't generate long PR/issue bodies -- users consistently ask agents to cut them down; 2-4 bullets max, no audit dumps
- Don't use separate shell calls for `git push` then `gh pr create` -- chain them with `&&`
- Don't manually write `Signed-off-by` in commit messages -- always use `git commit --signoff --gpg-sign` (`-s -S`) so the trailer matches `git config user.name` / `user.email` and the commit is cryptographically signed; DCO probot and signature verification both require exact identity match
- Don't use plain `gh pr view <number>` -- use `--json` form to avoid GraphQL errors about deprecated Projects (classic)

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub CLI Workflows

Multi-step recipes for common GitHub workflows. Quick command reference: [SKILL.md](../SKILL.md).

## Pre-Merge Checklist

Complete check before merging a PR:

```bash
PR_NUMBER=<number>

gh pr view $PR_NUMBER \
  --json number,title,state,isDraft,mergeable,reviewDecision,additions,deletions,changedFiles,labels,assignees,author,statusCheckRollup,reviews \
  --jq '{
    title, isDraft, mergeable, reviewDecision,
    author: .author.login,
    changes: "\(.additions)+/\(.deletions)- across \(.changedFiles) files",
    labels: [.labels[].name],
    checks: [.statusCheckRollup[] | {name: .name, conclusion: .conclusion}],
    reviews: [.reviews[] | {author: .author.login, state: .state}]
  }' \
  && gh pr checks $PR_NUMBER
```

Checklist:

1. `isDraft` is `false`
2. `mergeable` is `MERGEABLE`
3. `reviewDecision` is `APPROVED`
4. All CI checks show `pass`
5. PR title follows conventional commit format (enforced by `conventional-commit.yml`)

If any check fails, inspect the specific failure before proceeding.

## Debug Failed CI

Step-by-step pipeline debugging when CI fails on a PR:

### Step 1: Identify the Failed Run

```bash
# List recent failed runs for the current branch
gh run list --branch="$(git branch --show-current)" --status=failure

# Or list runs for a specific PR
gh run list --branch="$(gh pr view <number> --json headRefName -q .headRefName)"
```

### Step 2: View Failed Job Logs

```bash
RUN_ID=<run-id-from-step-1>

# Quick: show only failed job logs
gh run view $RUN_ID --log-failed
```

If the output is large, save to a file:

```bash
gh run view $RUN_ID --log-failed > /tmp/ci-failure.log
```

### Step 3: Reproduce Locally

Match the failed CI job to its local equivalent (see CI job table in SKILL.md). For full CI parity in a container:

```bash
make test-ci-container
```

When CI jobs are skipped (path filtering): the `changes` job skips format, typecheck, and unit-test when only non-source files are modified. To get CI to run: make a trivial Python change (e.g. docstring) or run `make check` and `make test` locally and note in the PR.

### Step 4: Fix and Re-run

After fixing locally:

```bash
git add -A && git commit -s -S -m "fix: resolve CI failure" && git push

# Or re-run failed jobs directly
gh run rerun $RUN_ID --failed
```

### Step 5: Monitor

```bash
# Watch the new run
gh run list --branch="$(git branch --show-current)" --limit=1
# Get the latest run ID, then:
gh run watch <new-run-id>
```

## Release Workflow

The release process uses the `release.yml` manual workflow dispatch.

### Step 1: Verify Readiness

```bash
git checkout main && git pull \
  && grep VERSION src/nemo_safe_synthesizer/package_info.py \
  && gh issue list --label "priority:high" --state open
```

### Step 2: Trigger the Release

```bash
# Dry run first (publishes to test PyPI only)
gh workflow run release.yml \
  -f release-ref=$(git rev-parse HEAD) \
  -f dry-run=true \
  -f create-gh-release=true \
  -f version-bump-branch=release/bump-version

# Monitor the run
gh run list --workflow=release.yml --limit=1
gh run watch <run-id>
```

### Step 3: Promote to Production

After verifying the dry run:

```bash
gh workflow run release.yml \
  -f release-ref=<tag-or-sha> \
  -f dry-run=false \
  -f create-gh-release=true \
  -f version-bump-branch=release/bump-version
```

### Step 4: Verify

```bash
# Check that the GitHub release was created
gh release view <tag>

# Verify PyPI (outside gh)
pip index versions nemo-safe-synthesizer
```

## Triage a New Issue

### Step 1: Review the Issue

```bash
gh issue view <number>
```

### Step 2: Label and Assign

```bash
gh issue edit <number> --add-label "bug" --add-label "priority:high" --add-assignee username
```

### Step 3: Link to Milestone (if applicable)

```bash
# List milestones to find the right one
gh api repos/{owner}/{repo}/milestones | jq '.[] | {number, title, due_on}'

# Set milestone via API
gh api repos/{owner}/{repo}/issues/<number> -f milestone=<milestone-number>
```

### Step 4: Acknowledge

```bash
gh issue comment <number> --body "Triaged. Assigned to @username, targeting milestone vX.Y."
```

## Review a PR Locally

End-to-end workflow: checkout a PR, run local checks, push fixes or leave a review.

### Step 1: Save Current Work and Checkout

```bash
# Stash any in-progress work
git stash

# Checkout the PR
gh pr checkout <number>
```

### Step 2: Understand the Changes

```bash
gh pr diff && gh pr view --json files -q '[.files[].path]'
```

### Step 3: Run Local Checks

```bash
make format && make check && make test
```

### Step 4a: Push Fixes (if contributing to the PR)

```bash
git add -A && git commit -s -S -m "fix: address lint/format issues" && git push
```

### Step 4b: Leave a Review (if just reviewing)

```bash
# Approve
gh pr review <number> --approve --body "LGTM, all checks pass locally"

# Request changes
gh pr review <number> --request-changes --body "Format check fails locally, see CI"
```

### Step 5: Return to Your Work

```bash
git checkout -
git stash pop
```

## Fetch and Address Review Comments

For "address PR comments" or "pull comments from PR N", use the **CLI helper** first. It returns a single JSON object (inline + top-level comments) and can post replies without the `gh` binary (GitHub API + `GITHUB_TOKEN`).

### Use the CLI helper (recommended)

From repo root (or skill dir). If `GITHUB_TOKEN` is not set, set it first: `export GITHUB_TOKEN=$(gh auth token)`.

```bash
# Optional: ensure token for API (helper uses GITHUB_TOKEN or gh auth token)
export GITHUB_TOKEN=$(gh auth token)

# Fetch all comments (single JSON: pr_number, repo, inline[], top_level[])
uv run --script .agents/skills/github-cli/scripts/gh_pr_helper.py -- comments <PR_NUMBER>

# Reply to an inline review comment (comment_id from the inline[].id in the JSON above)
uv run --script .agents/skills/github-cli/scripts/gh_pr_helper.py -- reply <COMMENT_ID> "Fixed in beefcafe"
# Or from stdin:
echo "Fixed in beefcafe" | uv run --script .agents/skills/github-cli/scripts/gh_pr_helper.py -- reply <COMMENT_ID> --reply-file -
# Or from a file (plain text or Markdown; content is sent as the comment body as-is):
uv run --script .agents/skills/github-cli/scripts/gh_pr_helper.py -- reply <COMMENT_ID> --reply-file path/to/reply.md
```

Reply file format: plain text or Markdown. The entire file contents become the comment body (no wrapper or front matter). Example `reply.md`:

```markdown
Fixed in commit `beefcafe`. The logic now uses the helper and the test was updated.
```

Workflow: you can draft the reply with the user in a file (e.g. `reply.md` or `pr-171-reply.md`), edit it until they’re happy, then run `reply <COMMENT_ID> --reply-file path/to/reply.md` to post it. No need to paste a long body on the command line.

- Omit `PR_NUMBER` to use the current branch’s open PR. Optional: `--repo OWNER/REPO`, `--token` / `-t`.
- In Agent use `required_permissions: ["all"]` (sandbox blocks network; see [Terminal docs](https://cursor.com/docs/agent/tools/terminal)).
- For batch-verifying many comment locations in parallel, consider an [explore subagent](https://cursor.com/docs/subagents).

### Alternative: raw gh / gh api

When the helper is not available or you need raw API JSON:

**Get all comments (inline + top-level):**

```bash
# Inline code review comments (not in gh pr view --json comments)
gh api repos/NVIDIA-NeMo/Safe-Synthesizer/pulls/<number>/comments

# Top-level PR conversation
gh pr view <number> --json comments -q '.comments[] | "\(.author.login): \(.body)"'
# Or raw JSON:
gh api repos/NVIDIA-NeMo/Safe-Synthesizer/issues/<number>/comments
```

**Reply to an inline review comment:**

```bash
gh issue comment <number> --body "Your reply here."   # top-level only
# Inline thread (comment-id from pulls/<number>/comments id field):
gh api repos/NVIDIA-NeMo/Safe-Synthesizer/pulls/comments/<comment-id>/replies -f body="Fixed in <commit-sha>"
```

### Address Comments in Code

Make fixes, commit with signoff:

```bash
git add -A && git commit -s -S -m "fix: address review feedback" && git push
```

### Update PR Body for Squash Merge

```bash
gh pr edit <number> --body "$(cat <<'EOF'
## Summary
- Updated summary reflecting all changes

## Test plan
- [x] Tests pass
EOF
)"
```


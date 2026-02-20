<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub CLI Workflows

Detailed multi-step recipes for common GitHub workflows. Referenced from [SKILL.md](../SKILL.md).

## Pre-Merge Checklist

Complete check before merging a PR:

```bash
PR_NUMBER=<number>

# Full status in one call (avoids Projects Classic GraphQL bug with plain `gh pr view`)
echo "=== PR Summary ==="
gh pr view $PR_NUMBER \
  --json number,title,state,isDraft,mergeable,reviewDecision,additions,deletions,changedFiles,labels,assignees,author,statusCheckRollup,reviews \
  --jq '{
    title, isDraft, mergeable, reviewDecision,
    author: .author.login,
    changes: "\(.additions)+/\(.deletions)- across \(.changedFiles) files",
    labels: [.labels[].name],
    checks: [.statusCheckRollup[] | {name: .name, conclusion: .conclusion}],
    reviews: [.reviews[] | {author: .author.login, state: .state}]
  }'

echo "=== CI Status ==="
gh pr checks $PR_NUMBER
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
gh run list --branch=$(git branch --show-current) --status=failure

# Or list runs for a specific PR
gh run list --branch=$(gh pr view <number> --json headRefName -q .headRefName)
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

Match the failed CI job to its local equivalent:

| CI Job | Local Command |
|--------|---------------|
| Format | `make format` or `bash tools/format/format.sh` |
| Lint | `make lint` or `bash tools/lint/ruff-lint.sh` |
| Typecheck | `bash tools/lint/run-ty-check.sh` |
| Unit Tests | `make test` or `make test-ci` |

For full CI parity in a container:

```bash
make test-ci-container
```

### Step 4: Fix and Re-run

After fixing locally:

```bash
git add -A && git commit -m "fix: resolve CI failure"
git push

# Or re-run failed jobs directly
gh run rerun $RUN_ID --failed
```

### Step 5: Monitor

```bash
# Watch the new run
gh run list --branch=$(git branch --show-current) --limit=1
# Get the latest run ID, then:
gh run watch <new-run-id>
```

## Release Workflow

The release process uses the `release.yml` manual workflow dispatch.

### Step 1: Verify Readiness

```bash
# Ensure main is clean and up to date
git checkout main && git pull

# Check the version in package_info.py
grep VERSION src/nemo_safe_synthesizer/package_info.py

# Confirm no open blockers
gh issue list --label "priority:high" --state open
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
# Add appropriate labels
gh issue edit <number> --add-label "bug"        # or "enhancement", "documentation"
gh issue edit <number> --add-label "priority:high"

# Assign to someone
gh issue edit <number> --add-assignee username
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
# See the diff
gh pr diff

# List changed files
gh pr view --json files | jq -r '.files[].path'

# Read the PR description
gh pr view <number>
```

### Step 3: Run Local Checks

```bash
# Format check
make format

# Lint
make lint

# Unit tests
make test
```

### Step 4a: Push Fixes (if contributing to the PR)

```bash
git add -A
git commit -m "fix: address lint/format issues"
git push
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

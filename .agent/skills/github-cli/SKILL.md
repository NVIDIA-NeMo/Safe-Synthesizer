---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: github-cli
description: "Interact with the Safe-Synthesizer GitHub repository using the gh CLI. Activate when users want to list or create pull requests, check out PRs, work on someone else's PR, check CI status, investigate workflow failures, view job logs, create or triage issues, check review and approval status, manage releases, or inspect repo metadata. Trigger keywords - pull request, PR, issue, workflow, CI, actions, failed job, job log, release, review, approve, CODEOWNERS, labels, milestone, checkout, gh, GitHub."
---

# GitHub CLI (gh)

## Detailed References

- [Full Reference](./references/full-reference.md) - Comprehensive `gh` reference for PRs, issues, Actions/CI, code review, releases, and repo metadata
- [Workflows](./references/workflows.md) - Pre-merge checklist, debug failed CI, release workflow, triage issues, review PRs locally

Note: The commands below are quick references. For comprehensive detail and multi-step workflows, see the references above.

## Shell Permissions

Always use `required_permissions: ["all"]` when running `gh` commands. Sandboxed environments fail with TLS certificate errors (`x509: OSStatus -26276`).

## Setup

```bash
brew install gh
gh auth login
gh auth status  # verify
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
```

## CI / Actions

```bash
# Check CI status for current branch
gh pr checks

# List workflow runs
gh run list --limit 10

# View a failed run
gh run view <run-id>

# View failed job logs
gh run view <run-id> --log-failed
```

## Issues

```bash
gh issue list
gh issue view <number>
gh issue create --title "Title" --body "Description"
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

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

## Pre-flight

Before any `gh` operation, verify the binary is available and authenticated:

```bash
# Verify gh is available (check PATH, then common install location)
which gh 2>/dev/null || ls ~/.local/bin/gh 2>/dev/null
# If found at ~/.local/bin/gh but not on PATH:
export PATH="$HOME/.local/bin:$PATH"
# Verify authentication
gh auth status
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

- Don't WebFetch GitHub Actions URLs for private repos -- use `gh run view` instead (auth required)
- Don't propose `gh pr create` without checking if a PR already exists first
- Don't generate long PR/issue bodies -- users consistently ask agents to cut them down

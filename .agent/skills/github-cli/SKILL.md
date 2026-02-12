---
name: github-cli
description: Interact with the Safe-Synthesizer GitHub repository using the gh CLI. Activate when users want to list or create pull requests, check out PRs, work on someone else's PR, check CI status, investigate workflow failures, view job logs, create or triage issues, check review and approval status, manage releases, or inspect repo metadata. Trigger keywords - pull request, PR, issue, workflow, CI, actions, failed job, job log, release, review, approve, CODEOWNERS, labels, milestone, checkout, gh, GitHub.
---

# GitHub CLI (`gh`)

Interact with the Safe-Synthesizer GitHub repository using the `gh` CLI for pull requests, issues, Actions/CI, code review, releases, and repo metadata.

## Shell Permissions

Always use `required_permissions: ["all"]` when running `gh` commands. While `["network"]` should suffice in theory, sandboxed environments often fail with TLS certificate errors (`x509: OSStatus -26276`). Using `["all"]` avoids this reliably.

## Setup

```bash
# macOS
brew install gh
gh auth login
```

```bash
# Linux (Debian/Ubuntu)
(type -p wget >/dev/null || sudo apt-get install wget -y) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt-get update \
  && sudo apt-get install gh -y
gh auth login
```

```bash
# Verify
gh auth status
```

## Pull Requests

### List PRs

```bash
# Open PRs in current repo
gh pr list

# Your open PRs
gh pr list --author=@me

# PRs awaiting your review
gh pr list --search "review-requested:@me"

# Filter by label or base branch
gh pr list --label "bug" --base main
```

### View PR Details

> **Known issue:** Plain `gh pr view <number>` can fail with a GraphQL error about
> "Projects (classic) being deprecated" if the org/repo has legacy project board data.
> Always prefer the `--json` form below, which avoids this entirely.

```bash
# Preferred: structured JSON view (avoids Projects Classic GraphQL bug)
gh pr view <number> --json number,title,state,url,headRefName,baseRefName,isDraft,mergeable,reviewDecision,statusCheckRollup,additions,deletions,changedFiles,reviews,labels,assignees,author \
  --jq '{
    number, title, state, url,
    branch: .headRefName, base: .baseRefName,
    draft: .isDraft, mergeable, reviewDecision,
    author: .author.login,
    additions, deletions, changedFiles,
    labels: [.labels[].name],
    assignees: [.assignees[].login],
    checks: [.statusCheckRollup[] | {name: .name, status: .status, conclusion: .conclusion}],
    reviews: [.reviews[] | {author: .author.login, state: .state}]
  }'

# Plain text summary (may fail -- see note above)
gh pr view <number>

# Subset of fields
gh pr view <number> --json title,state,author,reviews,statusCheckRollup,mergeable,labels

# View in browser
gh pr view <number> --web
```

### Create a PR

```bash
# Interactive
gh pr create

# Non-interactive with the project PR template structure
gh pr create --title "feat: add new feature" --body "$(cat <<'EOF'
<message>
EOF
)"

# Draft PR
gh pr create --draft --title "wip: early work" --body "Work in progress"
```

### Check Merge Readiness

```bash
# CI check status
gh pr checks <number>

# Full readiness (mergeable state, review decision, checks)
gh pr view <number> --json mergeable,reviewDecision,statusCheckRollup,isDraft,title | \
  jq '{title, mergeable, reviewDecision, isDraft, checks: [.statusCheckRollup[] | {name: .name, status: .status, conclusion: .conclusion}]}'
```

### Merge and Close

```bash
# Squash merge
gh pr merge <number> --squash

# Close without merging
gh pr close <number>
```

## Checking Out and Working on PRs

### Checkout a PR

```bash
# Checkout by PR number -- fetches the branch and switches to it
gh pr checkout <number>

# Checkout with a custom local branch name
gh pr checkout <number> -b my-local-branch

# Force re-fetch (useful if the PR was force-pushed)
gh pr checkout <number> --force
```

### View and Explore the PR

```bash
# View the diff against base
gh pr diff

# View the diff for specific files
gh pr diff -- src/nemo_safe_synthesizer/some_module.py

# See what changed in the PR (file list)
gh pr view --json files | jq -r '.files[].path'
```

### Push Changes to a Checked-Out PR

After `gh pr checkout`, the local branch tracks the PR author's remote branch. You can add commits and push directly **if the PR author enabled "Allow edits from maintainers"** (this is the default for PRs within the same repo).

```bash
# Make your changes, then:
git add -A
git commit -m "fix: address review feedback"
git push
```

If the PR is from a fork and edits are not allowed, push to your own branch and leave a comment:

```bash
git checkout -b my-fix-for-pr-<number>
git push -u origin my-fix-for-pr-<number>
gh pr comment <number> --body "Pushed suggested fixes to branch \`my-fix-for-pr-<number>\`"
```

### Sync a Checked-Out PR

```bash
# Pull latest changes from the PR author
git pull

# Or re-fetch from scratch
gh pr checkout <number> --force
```

### Stash and Return

Before switching away from your current work:

```bash
# Save current work
git stash

# Checkout the PR
gh pr checkout <number>

# ... review / make changes ...

# Return to your original branch
git checkout -
git stash pop
```

## Code Review

### Request Reviewers

```bash
# Add reviewers to a PR
gh pr edit <number> --add-reviewer user1,user2

# Add a team reviewer
gh pr edit <number> --add-reviewer NVIDIA-NeMo/safe-synthesizer-reviewers
```

### CODEOWNERS Reference

The project CODEOWNERS (`.github/CODEOWNERS`) assigns:

- **All files**: `@NVIDIA-NeMo/safe-synthesizer-maintainers`
- **`src/` and `tests/`**: `@NVIDIA-NeMo/safe-synthesizer-reviewers`
- **Critical files** (`pyproject.toml`, `uv.lock`, `SECURITY.md`, `LICENSE`, `.github/`): `@NVIDIA-NeMo/safe-synthesizer-maintainers`

### Review a PR

```bash
# Approve
gh pr review <number> --approve

# Request changes
gh pr review <number> --request-changes --body "Please fix the formatting issues"

# Leave a comment review
gh pr review <number> --comment --body "Looks good overall, minor suggestions inline"
```

### View Reviews

```bash
# List reviews on a PR
gh api repos/{owner}/{repo}/pulls/<number>/reviews | \
  jq '.[] | {user: .user.login, state: .state, submitted_at: .submitted_at}'

# View PR comments (review + general)
gh api repos/{owner}/{repo}/pulls/<number>/comments | \
  jq '.[] | {user: .user.login, path: .path, body: .body}'
```

## Issues

### List and View

```bash
# Open issues
gh issue list

# Filter by label
gh issue list --label "bug"

# Filter by assignee
gh issue list --assignee @me

# View details
gh issue view <number>

# View in browser
gh issue view <number> --web
```

### Create an Issue

The project provides issue templates: bug-report, feature-request, development-task.

```bash
# Interactive (will prompt for template selection)
gh issue create

# Non-interactive bug report
gh issue create --title "Bug: something broken" --body "Steps to reproduce..." --label "bug"

# With assignee
gh issue create --title "Task: implement X" --assignee @me --label "enhancement"
```

### Manage Issues

```bash
# Close
gh issue close <number>

# Reopen
gh issue reopen <number>

# Add labels
gh issue edit <number> --add-label "priority:high"

# Assign
gh issue edit <number> --add-assignee user1
```

## Actions / CI

The project has 6 GitHub Actions workflows:

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| CI Checks | `ci-checks.yml` | push to main, PRs, manual | format, lint, typecheck, unit-test |
| GPU jobs | `gpu-tests.yml` | push to main, PRs, manual | e2e job |
| Conventional Commit | `conventional-commit.yml` | PR title changes | Validates PR title format |
| Copyright Check | `copyright-check.yml` | push to main, release branches | NVIDIA copyright headers |
| DCO Assistant | `dco-assistant.yml` | PR events, comments | Developer Certificate of Origin |
| Release | `release.yml` | manual dispatch only | Build and publish to PyPI |
| Secrets Detector | `secrets-detector.yml` | PRs to main | Scans for leaked secrets |

### List Workflow Runs

```bash
# Recent runs across all workflows
gh run list

# Runs for a specific workflow
gh run list --workflow=ci-checks.yml

# Runs for the current branch
gh run list --branch=$(git branch --show-current)

# Failed runs only
gh run list --status=failure
```

### View a Run

```bash
# Summary
gh run view <run-id>

# View failed job logs directly
gh run view <run-id> --log-failed

# View full log for a specific job
gh run view <run-id> --log --job=<job-id>

# Watch a running workflow (polls until complete)
gh run watch <run-id>
```

### Re-run and Trigger

```bash
# Re-run only failed jobs
gh run rerun <run-id> --failed

# Re-run entire workflow
gh run rerun <run-id>

# Trigger manual workflow (e.g., release)
gh workflow run release.yml -f release-ref=<sha-or-tag> -f dry-run=true
```

### CI Jobs Map to Local Commands

When a CI job fails, reproduce locally with:

| CI Job | Local Command |
|--------|---------------|
| Format | `make format` or `bash tools/format/format.sh` |
| Lint | `make lint` or `bash tools/lint/ruff-lint.sh` |
| Typecheck | `bash tools/lint/run-ty-check.sh` |
| Unit Tests | `make test` or `make test-ci` |

Or run all CI checks in a container: `make test-ci-container`

## Releases

The primary release mechanism is the `release.yml` workflow dispatch. Direct `gh release create` is typically not used.

```bash
# List releases
gh release list

# View a specific release
gh release view <tag>

# Trigger a release (manual workflow dispatch)
gh workflow run release.yml \
  -f release-ref=<full-sha-or-tag> \
  -f dry-run=true \
  -f create-gh-release=true \
  -f version-bump-branch=<branch>
```

## Repo Metadata

### Labels

```bash
# List all labels
gh label list

# Create a label
gh label create "priority:high" --color "FF0000" --description "High priority item"
```

### Branch Protection and Milestones

```bash
# View branch protection rules for main
gh api repos/{owner}/{repo}/branches/main/protection

# List milestones
gh api repos/{owner}/{repo}/milestones | jq '.[] | {title, state, open_issues, due_on}'
```

## Common Workflows

For detailed multi-step recipes, see [workflows.md](workflows.md):

- **Pre-merge checklist** -- verify CI, reviews, conflicts, and conventional commit title
- **Debug failed CI** -- find the failed run, get logs, reproduce locally
- **Release workflow** -- trigger release.yml, monitor, verify
- **Triage a new issue** -- label, assign, link to milestone
- **Review a PR locally** -- checkout, run local checks, push fixes or leave review

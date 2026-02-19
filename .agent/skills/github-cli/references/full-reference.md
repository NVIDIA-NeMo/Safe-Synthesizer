<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub CLI (`gh`) -- Repo-Specific Reference

## Shell Permissions

Always use `required_permissions: ["all"]` when running `gh` commands. Sandboxed environments often fail with TLS certificate errors; `["all"]` avoids this.

## Viewing PRs

> **Known issue:** Plain `gh pr view <number>` can fail with a GraphQL error about
> "Projects (classic) being deprecated". Always use the `--json` form below.

```bash
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
```

## CODEOWNERS

Defined in `.github/CODEOWNERS`:

- **All files**: `@NVIDIA-NeMo/safe-synthesizer-maintainers`
- **`src/` and `tests/`**: `@NVIDIA-NeMo/safe-synthesizer-reviewers`
- **Critical files** (`pyproject.toml`, `uv.lock`, `SECURITY.md`, `LICENSE`, `.github/`): `@NVIDIA-NeMo/safe-synthesizer-maintainers`

```bash
gh pr edit <number> --add-reviewer NVIDIA-NeMo/safe-synthesizer-reviewers
```

## Actions / CI

The project has 7 GitHub Actions workflows:

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| CI Checks | `ci-checks.yml` | push to main, PRs, manual | format, lint, typecheck, unit-test |
| GPU jobs | `gpu-tests.yml` | push to main, PRs, manual | e2e job |
| Conventional Commit | `conventional-commit.yml` | PR title changes | Validates PR title format |
| Copyright Check | `copyright-check.yml` | push to main, release branches | NVIDIA copyright headers |
| DCO Assistant | `dco-assistant.yml` | PR events, comments | Developer Certificate of Origin |
| Release | `release.yml` | manual dispatch only | Build and publish to PyPI |
| Secrets Detector | `secrets-detector.yml` | PRs to main | Scans for leaked secrets |

### Investigating Failures

```bash
# Failed runs for current branch
gh run list --branch=$(git branch --show-current) --status=failure

# View failed job logs
gh run view <run-id> --log-failed

# Full log for a specific job
gh run view <run-id> --log --job=<job-id>

# Watch a running workflow
gh run watch <run-id>

# Re-run failed jobs only
gh run rerun <run-id> --failed
```

### CI Jobs Map to Local Commands

| CI Job | Local Command |
|--------|---------------|
| Format | `make format` or `bash tools/format/format.sh` |
| Lint | `make lint` or `bash tools/lint/ruff-lint.sh` |
| Typecheck | `bash tools/lint/run-ty-check.sh` |
| Unit Tests | `make test` or `make test-ci` |

Or run all CI checks in a container: `make test-ci-container`

## Releases

The primary release mechanism is the `release.yml` workflow dispatch:

```bash
gh workflow run release.yml \
  -f release-ref=<full-sha-or-tag> \
  -f dry-run=true \
  -f create-gh-release=true \
  -f version-bump-branch=<branch>
```

```bash
gh release list
gh release view <tag>
```

## Issue Templates

The project provides issue templates: `bug-report`, `feature-request`, `development-task`.

```bash
gh issue create  # Interactive, prompts for template
```

## Common Workflows

For multi-step recipes, see [workflows.md](workflows.md).

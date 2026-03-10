<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub Actions Workflows

This directory contains GitHub Actions workflows for CI/CD automation.

## Workflows Overview


| Workflow                                           | Trigger                               | Description                                          |
| -------------------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| [ci-checks.yml](ci-checks.yml)                     | Push to `main`, PRs, manual           | Format, typecheck, and unit tests (CPU)              |
| [gpu-tests.yml](gpu-tests.yml)                     | Push to `main`/`pull-request/*`, manual | GPU E2E tests (A100)                               |
| [conventional-commit.yml](conventional-commit.yml) | PRs                                   | Validates PR titles follow conventional commit format |
| [copyright-check.yml](copyright-check.yml)         | Push to `main`/`pull-request/*`        | Validates NVIDIA copyright headers on Python files   |
| [docs.yml](docs.yml)                               | Push to `main` (docs paths)           | Builds and deploys documentation to GitHub Pages     |
| [internal-release.yml](internal-release.yml)       | Tag push (`v[0-9]*`), manual dispatch | Builds and publishes wheel to Artifactory or PyPI    |
| [release.yml](release.yml)                         | Manual dispatch                       | Builds and publishes package to PyPI (production)    |
| [secrets-detector.yml](secrets-detector.yml)       | PRs                                   | Scans for accidentally committed secrets             |


## Pull Request Testing (copy-pr-bot)

GPU tests (`gpu-tests.yml`) run on NVIDIA self-hosted runners, which block `pull_request`-triggered jobs. They use the [copy-pr-bot](https://docs.gha-runners.nvidia.com/platform/apps/copy-pr-bot/) pattern instead:

1. When a PR is opened by a trusted user with trusted changes, `copy-pr-bot` automatically copies the code to a `pull-request/<number>` branch
2. The push to `pull-request/<number>` triggers the GPU workflow
3. Untrusted PRs require a vetter to comment `/ok to test <SHA>` before GPU tests run
4. Draft PRs do not auto-sync (`auto_sync_draft: false`), saving GPU resources

Configuration: [`.github/copy-pr-bot.yaml`](../copy-pr-bot.yaml)

CPU checks (`ci-checks.yml`) run on GitHub-hosted `ubuntu-latest` runners and use standard `pull_request` triggers.

## Workflow Diagram

```mermaid
flowchart LR
    subgraph triggers [Triggers]
        push[Push to main]
        cpb[copy-pr-bot push to pull-request/*]
        pr[Pull Request event]
        manual[Manual Dispatch]
    end

    subgraph ci [CI Checks - GitHub-hosted runners]
        changes_ci[Detect Changes]
        format[Format]
        typecheck[Typecheck]
        unit[Unit Tests]
        ci_status[CI Status]
        changes_ci --> format & typecheck & unit
        format & typecheck & unit --> ci_status
    end

    subgraph gpu [GPU Tests - on-prem runners]
        changes_gpu[Detect Changes]
        e2e[GPU E2E Tests]
        gpu_status[GPU CI Status]
        changes_gpu --> e2e --> gpu_status
    end

    subgraph compliance [Compliance Workflows]
        conventional[Conventional Commit]
        secrets[Secrets Detector]
        copyright[Copyright Check]
    end

    subgraph release [Release Workflow]
        buildWheel[Build Wheel]
        publishPyPI[Publish to PyPI]
        ghRelease[GitHub Release]
        slackNotify[Slack Notification]
    end

    subgraph internalRelease [Internal Release]
        buildWheelInt[Build Wheel]
        publishArtifactory[Publish to Artifactory/PyPI]
    end

    push --> ci & gpu
    cpb --> gpu & copyright
    pr --> ci & conventional & secrets
    manual --> release
    tag[Tag push v[0-9]*] --> internalRelease

    buildWheel --> publishPyPI --> ghRelease --> slackNotify
    buildWheelInt --> publishArtifactory

    conventional -.->|reuses| FW-CI-templates
    secrets -.->|reuses| FW-CI-templates
    copyright -.->|reuses| FW-CI-templates
    release -.->|reuses| FW-CI-templates
```

## CI Checks Workflow

The `ci-checks.yml` workflow runs on every push to `main` and on pull requests. Every check step calls a `make` target so the Makefile is the single source of truth for how each check runs.

| Job | `make` target | What it checks |
|---|---|---|
| Format | `format-check` | `ruff format --check` + `ruff check` + SPDX copyright headers |
| Format (lock) | `lock-check` | `uv.lock` matches `pyproject.toml` |
| Typecheck | `typecheck` | `ty check` (excludes per `pyproject.toml [tool.ty.src]`) |
| Unit Tests | `test-ci` | pytest with coverage |

The `changes` detection job (using `dorny/paths-filter`) skips downstream jobs entirely when only non-source files are modified. Within each job, all tracked files are checked -- `ruff` and `ty` are fast enough for this to take seconds. The CI Status aggregation job is the single required check for branch protection.

To replicate CI locally:

```bash
make check       # format-check + typecheck
make lock-check  # verify uv.lock
make test        # unit tests
```

All jobs run on `ubuntu-latest` (GitHub-hosted).

## GPU Tests Workflow

The `gpu-tests.yml` workflow runs on pushes to `main` and `pull-request/*` branches (via copy-pr-bot):

- GPU E2E Tests: Runs end-to-end tests on `linux-amd64-gpu-a100-latest-1` (A100) with a 60-minute job timeout and 45-minute step timeout
- GPU CI Status: Aggregation job -- single required check for branch protection

### Runners

| Workflow | Job | Runner Label | Type |
| --- | --- | --- | --- |
| CI Checks | All jobs | `ubuntu-latest` | GitHub-hosted |
| GPU Tests | GPU E2E Tests | `linux-amd64-gpu-a100-latest-1` | NVIDIA self-hosted GPU (A100) |
| GPU Tests | Detect Changes, GPU CI Status | `linux-amd64-cpu4` | NVIDIA self-hosted CPU (4-core) |
| Dev Wheel | All jobs | `linux-amd64-cpu4` | NVIDIA self-hosted CPU (4-core) |
| Internal Release | All jobs | `linux-amd64-cpu4` | NVIDIA self-hosted CPU (4-core) |

### Coverage

Coverage reports are uploaded as artifacts from both workflows.

## Compliance Workflows

### Conventional Commit

PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test changes
- `build:` - Build system changes
- `ci:` - CI configuration changes
- `chore:` - Maintenance tasks
- `revert:` - Reverts
- `cp:` - Cherry-picks

### DCO Assistant

Contributors must sign the Developer Certificate of Origin. Sign by adding to commit messages:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Or comment on the PR: `I have read the DCO Document and I hereby sign the DCO`

### Secrets Detector

Scans PRs for accidentally committed secrets. False positives can be added to `.github/workflows/config/.secrets.baseline`.

### Copyright Check

Validates that Python files have proper NVIDIA copyright headers.

## Internal Release Workflow

The `internal-release.yml` workflow builds a wheel and publishes it to NVIDIA Artifactory or PyPI.

### Triggers

**Tag push (automatic):** Pushing a `v[0-9]*` tag (e.g. `git tag v0.2.0 && git push --tags`) automatically builds and publishes to Artifactory. This is the primary release mechanism.

**Manual dispatch:** Go to Actions > Internal Release and run with:
- `release-ref`: Branch, tag, or commit SHA to build (defaults to `main`)
- `publish-target`: `artifactory` (default) or `pypi`

### How to Publish Internally

Tag-based (recommended):

```bash
git tag v0.2.0
git push --tags
```

This triggers the workflow automatically and publishes to Artifactory.

Via GitHub Actions (manual):

1. Go to Actions > Internal Release
2. Click Run workflow
3. Enter the branch, tag, or commit SHA to build
4. Select publish target (`artifactory` or `pypi`)

Requires `ARTIFACTORY_USERNAME`, `ARTIFACTORY_TOKEN`, and `ARTIFACTORY_INTERNAL_URL` secrets for Artifactory; `TWINE_USERNAME` and `TWINE_PASSWORD` for PyPI.

Locally (via Makefile):

Add the required env vars to your `.local.envrc` (git-ignored):

```bash
export TWINE_REPOSITORY_URL=<artifactory-repo-url>
export TWINE_USERNAME=<your-username>
export TWINE_PASSWORD=<your-api-key>
```

Then run:

```bash
# Build wheel only
make build-wheel

# Build and publish to Artifactory
make publish-internal
```

## Release Workflow (Production)

The production release workflow uses the [FW-CI-templates `_release_library.yml`](https://github.com/NVIDIA-NeMo/FW-CI-templates) reusable workflow to publish to PyPI.

### How to Release

this is placeholder information until we do a real release. will update then.

1. Go to Actions > Release NeMo Safe Synthesizer
2. Click Run workflow
3. Fill in the required inputs:
  - `release-ref`: Full SHA or tag of the commit to release
  - `dry-run`: Set to `false` for production release (publishes to PyPI)
  - `create-gh-release`: Whether to create a GitHub release
  - `version-bump-branch`: Branch to push the version bump PR (usually `main`)

### Release Process

The workflow performs the following steps:

1. Dry-run build - Validates the wheel can be built
2. Version bump - Creates a PR to bump the version in `package_info.py`
3. Build wheel - Builds the production wheel
4. Publish to PyPI - Uploads to PyPI (or test PyPI for dry runs)
5. Create GitHub release - Creates a tagged release with changelog
6. Notify - Sends Slack notification

### Version Management

Version is managed in `[src/nemo_safe_synthesizer/package_info.py](../../src/nemo_safe_synthesizer/package_info.py)`:

```python
MAJOR = 0
MINOR = 1
PATCH = 0
PRE_RELEASE = ""
BUILD = 1
DEV_RELEASE = False
```

The release workflow automatically bumps the PATCH version (or PRE_RELEASE for release candidates).

## Required Secrets

The following secrets must be configured in GitHub repository settings:


| Secret                   | Purpose                      |
| ------------------------ | ---------------------------- |
| `TWINE_USERNAME`         | PyPI username                |
| `TWINE_PASSWORD`         | PyPI API token               |
| `SLACK_WEBHOOK_ADMIN`    | Slack admin notifications    |
| `SLACK_RELEASE_ENDPOINT` | Slack release notifications  |
| `PAT`                    | GitHub Personal Access Token |
| `SSH_KEY`                | GPG signing key              |
| `SSH_PWD`                | GPG key passphrase           |
| `BOT_KEY`                | GitHub App private key       |
| `ARTIFACTORY_USERNAME`     | NVIDIA Artifactory username  |
| `ARTIFACTORY_TOKEN`       | NVIDIA Artifactory API key   |
| `ARTIFACTORY_INTERNAL_URL`| NVIDIA Artifactory repo URL  |



| Variable | Purpose       |
| -------- | ------------- |
| `BOT_ID` | GitHub App ID |


## Reusable Workflows

All compliance and release workflows reuse templates from [NVIDIA-NeMo/FW-CI-templates](https://github.com/NVIDIA-NeMo/FW-CI-templates) (pinned to `v0.66.6`):

- `_semantic_pull_request.yml` - Conventional commit validation
- `_secrets-detector.yml` - Secrets scanning
- `_copyright_check.yml` - Copyright header validation
- `_release_library.yml` - Full release automation

## Configuration Files


| File                                              | Purpose                              |
| ------------------------------------------------- | ------------------------------------ |
| `config/.secrets.baseline`                        | False positives for secrets detector |
| `../../.python-version`                           | Python version for uv packaging      |
| `../../src/nemo_safe_synthesizer/package_info.py` | Version information                  |

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GitHub Actions Workflows

This directory contains GitHub Actions workflows for CI/CD automation.

## Workflows Overview

All workflows that use `.github/actions/setup-python-env` now default to the version in `../../.python-version`. Set the action input `python-version` only when a job intentionally needs an override.

| Workflow                                           | Trigger                     | Description                                                                                                |
| -------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [ci-checks.yml](ci-checks.yml)                     | Push to `main`, PRs, manual | Format, typecheck, unit tests, and CPU smoke tests                                                         |
| [gpu-tests.yml](gpu-tests.yml)                     | Nightly, manual             | GPU smoke tests (required) and E2E tests                                                                   |
| [container-build.yml](container-build.yml)         | `v*`, manual                 | Builds the extra-driven GPU container image and publishes GHCR tags for release tags                       |
| [conventional-commit.yml](conventional-commit.yml) | PRs                         | Validates PR titles follow conventional commit format                                                      |
| [docs.yml](docs.yml)                               | Push to `main` (docs paths) | Publishes `main` docs as the `latest` GitHub Pages version                                                 |
| [release.yml](release.yml)                         | Push tags to `v*`           | Builds and publishes package to Test PyPI/PyPI, creates a GitHub release, and publishes versioned docs     |
| [secrets-detector.yml](secrets-detector.yml)       | PRs                         | Scans for accidentally committed secrets                                                                   |

## Pull Request Testing

GPU tests on PRs are currently disabled due to internal constraints. `gpu-tests.yml` has its `push` trigger commented out, so it runs only on the nightly schedule or manual `workflow_dispatch`.

GPU tests (`gpu-tests.yml`) run on NVIDIA self-hosted runners, which block `pull_request`-triggered jobs. When PR GPU testing is re-enabled, use the [copy-pr-bot](https://docs.gha-runners.nvidia.com/platform/apps/copy-pr-bot/) pattern:

1. When a PR is opened by a trusted user with trusted changes, `copy-pr-bot` automatically copies the code to a `pull-request/<number>` branch
2. The push to `pull-request/<number>` triggers the GPU workflow
3. Untrusted PRs require a vetter to comment `/ok to test <SHA>` before GPU tests run
4. Draft PRs do not auto-sync (`auto_sync_draft: false`), saving GPU resources

Configuration: [`.github/copy-pr-bot.yaml`](../copy-pr-bot.yaml)

CPU checks (`ci-checks.yml`) run on GitHub-hosted `ubuntu-latest` runners and use standard `pull_request` triggers.

### On-demand GPU test runs for PRs

This path is disabled while the `push` trigger in `gpu-tests.yml` is commented out. When it is re-enabled, comment `/sync` on the PR to trigger a GPU test run without waiting for auto-sync. copy-pr-bot will push the current HEAD to `pull-request/<number>`, fire `gpu-tests.yml`, and post the `GPU CI Status` check result back to the PR -- the same check as the automatic trigger.

When this path is re-enabled, use `/sync` when:

- The PR is a draft (auto-sync is disabled for drafts)
- You want to re-run after a flaky failure without pushing a new commit
- You want a GPU test result before marking the PR ready for review

## Workflow Diagram

```mermaid
flowchart LR
    subgraph triggers [Triggers]
        push[Push to main]
        schedule[Nightly Schedule]
        pr[Pull Request event]
        manual[Manual Dispatch]
    end

    subgraph ci [CI Checks - GitHub-hosted runners]
        changes_ci[Detect Changes]
        format[Format]
        typecheck[Typecheck]
        unit[Unit Tests]
        smoke_cpu[Smoke Tests]
        ci_status[CI Status]
        changes_ci --> unit & smoke_cpu
        format & typecheck & unit & smoke_cpu --> ci_status
    end

    subgraph gpu [GPU Tests - on-prem runners]
        changes_gpu[Detect Changes]
        gpu_smoke[GPU Smoke Tests]
        e2e[GPU E2E Tests]
        gpu_status[GPU CI Status]
        changes_gpu --> gpu_smoke & e2e
        gpu_smoke & e2e --> gpu_status
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

    subgraph containers [Container Build]
        buildContainer[Build cu129/cu130 Images]
        publishGhcr[Publish GHCR Tags]
    end

    subgraph internalRelease [Internal Release]
        buildWheelInt[Build Wheel]
        publishArtifactory[Publish to Artifactory/PyPI]
    end

    push --> ci
    schedule --> gpu
    manual --> ci & gpu
    pr --> ci & conventional & secrets
    tag[Tag push v[0-9]*] --> release & containers

    buildWheel --> publishPyPI --> ghRelease --> slackNotify
    buildContainer --> publishGhcr
    buildWheelInt --> publishArtifactory

    conventional -.->|reuses| FW-CI-templates
    secrets -.->|reuses| FW-CI-templates
```

## CI Checks Workflow

The `ci-checks.yml` workflow runs on every push to `main` and on pull requests. Every check step calls a mise task. Declarative tasks live in `.mise/tasks/*.toml`; bash-heavy tasks are executable file tasks under `.mise/tasks/`.

| Job | mise task | What it checks |
| --- | --- | --- |
| Format | `format-check` | `dprint check` for TOML + `ruff format --check` + `ruff check` + SPDX copyright headers |
| Format (lock) | `lock-check` | generated CUDA metadata matches `cuda_deps.toml` + `uv.lock` matches `pyproject.toml` |
| Typecheck | `typecheck` | `ty check` (excludes per `pyproject.toml [tool.ty.src]`) |
| Unit Tests | `test:ci` | pytest with coverage (excludes slow, e2e, gpu, smoke) |
| Smoke Tests | `test:smoke` | CPU smoke tests (training/generation hot paths, tiny models) |

The `changes` detection job uses `dorny/paths-filter` to decide which test jobs run on push and pull request events. Format and typecheck intentionally do not depend on `changes`; they are ungated and run on every push, pull request, and manual dispatch. Unit tests run when any tracked source, docs source, test, dependency, CI, or mise task path changes. Smoke tests run when source, test, `pytest.ini`, or dependency paths change.

On manual dispatch, `changes` is intentionally skipped and the test jobs explicitly bypass that skipped dependency. Manual dispatch runs unit tests and CPU smoke tests even when there is no changed-file signal to inspect.

Docs source paths include `docs/*.py`, `docs/**/*.py`, and `mkdocs.yml`. These paths trigger unit tests through the aggregate `any` output, but do not trigger CPU smoke tests. The CI Status aggregation job is the single required check for branch protection.

To replicate CI locally:

```bash
mise run check        # format-check + typecheck
mise run lock-check   # verify generated CUDA metadata and uv.lock
mise run test:ci      # CI unit tests with coverage selectors
mise run test:smoke   # CPU smoke tests
```

All jobs run on `ubuntu-latest` (GitHub-hosted).

## GPU Tests Workflow

The `gpu-tests.yml` workflow runs nightly at 02:00 UTC, and can also be triggered manually via `workflow_dispatch`. Manual dispatch includes a `suite` dropdown with `all`, `smoke`, and `e2e` options. The `push` trigger for `pull-request/*` branches is currently commented out due to internal blockers, so PRs do not automatically produce GPU status checks. We expect to re-enable that path as soon as those blockers are resolved. There are several key jobs:

- GPU Smoke Tests: runs GPU-marked unit tests, followed by staged train-only, generation, resume, structured generation, timeseries, and SmolLM2 smoke tests. Required for merge when the workflow is part of branch protection.
- GPU E2E Tests: End-to-end tests on a gpu runner with a 210-minute job timeout and 190-minute step timeout. Informational -- failures produce a warning but don't block merge.
- GPU CI Status: Aggregation job for the GPU workflow. It is not currently a live branch-protection requirement while PR GPU runs are disabled; when re-enabled, it is intended to be the required GPU check. It fails if smoke tests fail and warns if E2E tests fail.

The `changes` (Detect Changes) job is skipped on `workflow_dispatch`. GPU jobs use `always()` in their job conditions so manual runs can bypass the skipped dependency and run the selected suite. On scheduled runs, `changes` gates GPU jobs with the `src_test_deps` output, which is true for source, test, `pytest.ini`, dependency, or CI workflow/action changes.

GPU jobs use `.github/actions/setup-gpu-test-env` for shared GPU setup. Following the release wheel verification's clean-room pattern, the action builds the wheel and installs `[cu129,engine]` plus test tooling with uv configuration and project sources disabled. Dependencies resolve from PyPI and the required CUDA wheel indexes without consulting `uv.lock` before GPU availability is checked.

To trigger manually from the CLI (produces a run but not a PR status check):

```bash
gh workflow run gpu-tests.yml --ref <branch-name> -f suite=all
gh workflow run gpu-tests.yml --ref <branch-name> -f suite=smoke
gh workflow run gpu-tests.yml --ref <branch-name> -f suite=e2e
```

PR status-check GPU runs are currently disabled while the workflow `push` trigger is commented out due to internal blockers. When that path is re-enabled, `/sync` will provide the PR-status flow described in [On-demand GPU test runs for PRs](#on-demand-gpu-test-runs-for-prs).

### Runners

Internal runners and projects are defined in an internal repo, `nv-gha-runners/enterprise-runner-configuration`.

| Workflow | Job | Runner Label | Type |
| --- | --- | --- | --- |
| CI Checks | All jobs | `ubuntu-latest` | GitHub-hosted |
| GPU Tests | GPU Smoke Tests | `linux-amd64-gpu-a100-latest-1` | NVIDIA self-hosted GPU |
| GPU Tests | GPU E2E Tests | `linux-amd64-gpu-a100-latest-1` | NVIDIA self-hosted GPU |
| GPU Tests | Detect Changes, GPU CI Status | `linux-amd64-cpu4` | NVIDIA self-hosted CPU (4-core) |
| Dev Wheel | All jobs | `linux-amd64-cpu4` | NVIDIA self-hosted CPU (4-core) |
| Internal Release | All jobs | `linux-amd64-cpu4` | NVIDIA self-hosted CPU (4-core) |

### Coverage

Coverage reports are uploaded as artifacts from the unit test job.

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

## Release Workflow (Production)

The production release workflow verifies and publishes the wheel to Test PyPI
and PyPI, creates a GitHub release, and publishes versioned documentation for
final releases.

### How to Release

Follow the release preparation, candidate validation, promotion, and
post-release procedures in the [contributor guide](../../CONTRIBUTING.md#releasing).
The guide documents the read-only `release:prepare` helper and the supported
tag formats.

### Release Process

The workflow performs the following steps:

1. Build wheel - Builds the production wheel
2. Verify the wheel installs in a clean end-user container
3. Push to test PyPI
4. Publish to PyPI - Uploads to PyPI
5. Create GitHub release
6. Publish versioned documentation for final releases

## Reusable Workflows

Compliance workflows reuse templates from [NVIDIA-NeMo/FW-CI-templates](https://github.com/NVIDIA-NeMo/FW-CI-templates), pinned to immutable commit SHAs in the workflow files:

- `_semantic_pull_request.yml` - Conventional commit validation
- `_secrets-detector.yml` - Secrets scanning
- `_copyright_check.yml` - Copyright header validation
- `_release_library.yml` - Full release automation

Run the changed-file secrets scan locally with the repository tool:

```bash
uv run --no-project --with detect-secrets==1.5.0 -- \
  bash tools/secrets-detector-changed-files.sh origin/main
```

## Configuration Files

| File | Purpose |
| --- | --- |
| `config/.secrets.baseline` | False positives for secrets detector |
| `../../.python-version` | Python version source for CI |
| `../../src/nemo_safe_synthesizer/package_info.py` | Version information |

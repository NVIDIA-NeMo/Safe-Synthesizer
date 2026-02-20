# Contributing to NeMo Safe Synthesizer

Thank you for your interest in contributing to NeMo Safe Synthesizer! This document provides guidelines and information for contributors.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Repository Settings](#repository-settings)
  - [Branch Naming Convention](#branch-naming-convention)
  - [Conventional Commits](#conventional-commits)
  - [Semantic Versioning for Tags](#semantic-versioning-for-tags)
  - [Branch Protection](#branch-protection)
- [Pull Request Process](#pull-request-process)
- [Issues and Discussions](#issues-and-discussions)
- [Developer Certificate of Origin](#developer-certificate-of-origin)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- [gh](https://cli.github.com/) - GitHub CLI (optional, for PR workflows)

> **Note:** Other tools like [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), and [ty](https://github.com/astral-sh/ty) are installed automatically by `make bootstrap-tools`.

### Setup

1. Fork the repository on GitHub
2. Clone your fork:
  ```bash
   git clone https://github.com/<your-username>/safe-synthesizer.git
   cd safe-synthesizer
  ```
3. Set up the development environment:
  ```bash
   # Install development tools (uv, ruff, ty, yq, etc.)
   make bootstrap-tools

   # Install Python dependencies (choose one)
   make bootstrap-nss cpu    # CPU-only (macOS or Linux without GPU)
   make bootstrap-nss cuda   # CUDA 12.8 (Linux with NVIDIA GPU)
   make bootstrap-nss engine # Engine dependencies only
   make bootstrap-nss dev    # Minimal dev dependencies only
  ```
4. Add the upstream remote:
  ```bash
   git remote add upstream https://github.com/NVIDIA-NeMo/safe-synthesizer.git
  ```

### NMP Integration

NeMo Safe Synthesizer is a standalone package. Changes flow into NMP via Artifactory publishing and vendor packaging. See the [NMP Integration](README.md#nmp-integration) section of the README for details on publishing, SDK vendoring, and local development workflows.

## Repository Settings

This repository uses GitHub Rulesets to enforce consistent contribution standards. These rules are automatically enforced—you don't need to configure anything, but you should understand them to contribute successfully.

### Branch Naming Convention

All branches (except `main`) must follow this naming pattern:

```
<author>/<description>
<author>/<issue-id>-<description>
<author>/<type>/<description>
<author>/<type>/<issue-id>-<description>
```

Rules:

- `<author>`: Your GitHub username (lowercase, alphanumeric, hyphens allowed)
- `<issue-id>`: Optional GitHub issue number prefix (e.g., `123-`)
- `<description>`: Brief description (lowercase, alphanumeric, hyphens)
- `<type>`: Optional category prefix

Valid types: `feature`, `bugfix`, `hotfix`, `release`, `docs`, `chore`, `test`

Examples:


| Branch Name                       | Valid               |
| --------------------------------- | ------------------- |
| `jsmith/add-login-feature`        | ✅                   |
| `jsmith/123-add-login-feature`    | ✅                   |
| `jsmith/feature/123-add-login`    | ✅                   |
| `aagonzales/bugfix/456-fix-crash` | ✅                   |
| `dev-team/docs/update-readme`     | ✅                   |
| `feature/add-login`               | ❌ Missing author    |
| `JSmith/123-Add-Login`            | ❌ Must be lowercase |


### Conventional Commits

All commits merged to `main` must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>
```

or without scope:

```text
<type>: <description>
```

Rules:

- `<type>`: Required, must be one of the valid types below
- `<scope>`: Optional, indicates the area of the codebase affected
- `<description>`: Required, brief description (max 100 characters)
- Add `!` after type/scope for breaking changes

Valid types:


| Type       | Description                                      |
| ---------- | ------------------------------------------------ |
| `feat`     | New feature                                      |
| `fix`      | Bug fix                                          |
| `docs`     | Documentation changes                            |
| `style`    | Code style changes (formatting, no logic change) |
| `refactor` | Code refactoring (no feature or fix)             |
| `perf`     | Performance improvements                         |
| `test`     | Adding or updating tests                         |
| `build`    | Build system or dependencies                     |
| `ci`       | CI/CD configuration                              |
| `chore`    | Maintenance tasks                                |
| `revert`   | Reverting previous commits                       |


Examples:


| Commit Message                            | Valid                    |
| ----------------------------------------- | ------------------------ |
| `feat: add user authentication`           | ✅                        |
| `fix(auth): resolve token expiration bug` | ✅                        |
| `docs: update API documentation`          | ✅                        |
| `chore(deps)!: bump major dependencies`   | ✅ Breaking change        |
| `Added new feature`                       | ❌ Missing type           |
| `fix - resolve bug`                       | ❌ Wrong format           |
| `FIX: resolve bug`                        | ❌ Type must be lowercase |


> Since we use squash merging, your PR title should follow this format as it becomes the commit message.

### Semantic Versioning for Tags

Release tags must follow [Semantic Versioning](https://semver.org/):

```text
MAJOR.MINOR.PATCH[-prerelease][+build]
```

Examples:


| Tag                    | Valid           |
| ---------------------- | --------------- |
| `1.0.0`                | ✅               |
| `2.1.3`                | ✅               |
| `1.0.0-alpha`          | ✅               |
| `1.0.0-beta.1`         | ✅               |
| `1.0.0-rc.1+build.123` | ✅               |
| `v1.0.0`               | ❌ No `v` prefix |
| `release-1.0`          | ❌ Wrong format  |


### Branch Protection

The `main` branch has the following protections:


| Rule                            | Setting     |
| ------------------------------- | ----------- |
| Required approvals              | 1           |
| Code owner review               | Required    |
| Dismiss stale reviews           | No          |
| Require conversation resolution | Yes         |
| Linear history                  | Required    |
| Force pushes                    | Blocked     |
| Deletions                       | Blocked     |
| Merge strategy                  | Squash only |


## Pull Request Process

1. Create an issue first (if one doesn't exist) to discuss the change
2. Create a branch following the [naming convention](#branch-naming-convention):
  ```bash
   git checkout -b <username>/<issue-id>-<description>
  ```
3. Make your changes and commit using [conventional commits](#conventional-commits)
4. Run tests locally:
  ```bash
   make test
  ```
5. Push your branch:
  ```bash
   git push origin <your-branch>
  ```
6. Open a Pull Request using the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
7. Address review feedback — reviewers from [CODEOWNERS](.github/CODEOWNERS) will be automatically assigned
8. Merge — once approved, your PR will be squash-merged and the branch auto-deleted

### CODEOWNERS

- All `src` and `test` files: `@NVIDIA-NeMo/safe-synthesizer-reviewers` 
- all remaining files: (`pyproject.toml`, `uv.lock`, `SECURITY.md`, `LICENSE`, `.github/, etc.`): `@NVIDIA-NeMo/safe-synthesizer-maintainers`

## Issues and Discussions

### Issue Templates

We provide structured issue templates:

- Bug Report — Report a bug with reproduction steps
- Feature Request — Propose a new feature
- Development Task — Track internal development work

### Questions

For general questions, please use [GitHub Discussions](https://github.com/NVIDIA-NeMo/safe-synthesizer/discussions) instead of opening an issue.

## Developer Certificate of Origin

All contributions must be signed off to certify that you have the right to submit the code. This is done by adding a `Signed-off-by` line to your commit messages.

Sign off your commits:

```bash
git commit -s -m "feat: add new feature"
```

This adds a line like:

```text
Signed-off-by: Your Name <your.email@example.com>
```

By signing off, you certify the [Developer Certificate of Origin](DCO):

> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications...

See the full [DCO](DCO) file for details.

## Testing

### Running Tests

```bash
# Run unit tests (excludes slow and e2e)
make test

# Run all tests including slow tests (excludes e2e)
make test-slow

# Run SDK-related tests (config, sdk, cli, api)
make test-sdk-related

# Run GPU integration tests (requires CUDA)
make test-gpu-integration

# Run end-to-end tests (requires CUDA)
make test-e2e

# Run CI tests locally in a Linux container (Docker/Podman)
make test-ci-container

# Run specific test files directly
uv run pytest tests/cli/test_run.py
```

### Test Requirements

Before submitting a PR:

- All existing tests pass (`make test`)
- New features include tests
- Bug fixes include regression tests

## Code Style

### Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for code formatting and import sorting. The formatter runs on changed files against the `main` branch.

```bash
# Format code and sort imports
make format
```

### Linting and Type Checking

We use [Ruff](https://docs.astral.sh/ruff/) for linting and [ty](https://github.com/astral-sh/ty) for type checking. Both run on changed files against `main`.

```bash
# Run ruff linter (with auto-fix) and ty type checker
make lint
```

### Pre-commit Hooks

We recommend setting up pre-commit hooks to catch formatting, linting, and type issues before committing:

```bash
prek install
```

This installs hooks that run Ruff (format + lint), ty type checking, and uv lock verification on each commit.

## Documentation

This project uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) for its documentation site, hosted at <https://nvidia-nemo.github.io/Safe-Synthesizer/>.

### Local Preview

Documentation dependencies are included in the `dev` bootstrap profile. If you already ran `make bootstrap-nss dev` (or `cpu`/`cuda`), you're set. Otherwise install them directly:

```bash
uv sync --group docs
```

Start a local server with live reload:

```bash
make docs-serve
# Browse to http://127.0.0.1:8000
```

Build the static site (output in `site/`):

```bash
make docs-build
```

### Directory Layout

All documentation lives under `docs/`. The structure follows the [Diataxis](https://diataxis.fr/) framework:

| Directory | Content type | Examples |
| --- | --- | --- |
| `getting-started/` | Tutorials | Installation, quick start |
| `user-guide/` | How-tos & reference | CLI, configuration, SDK |
| `architecture/` | Explanations | Design decisions |
| `reference/` | API reference | Auto-generated (see below) |
| `blog/` | Dev notes | Release notes, design posts |

### Adding or Editing a Page

1. Create or edit the `.md` file under the appropriate `docs/` subdirectory.
2. Add the page to the `nav:` section of `mkdocs.yml` so it appears in the sidebar.
3. Run `make docs-serve` and verify the page renders correctly.

### MkDocs Material Features

The site configuration (`mkdocs.yml`) enables several useful Markdown extensions:

- **Admonitions** -- callout boxes (`!!! note`, `!!! warning`, `??? tip` for collapsible)
- **Content tabs** -- tabbed content blocks (`=== "Python SDK"` / `=== "CLI"`)
- **Code blocks** -- syntax highlighting, line numbers, copy button, and annotations
- **Mermaid diagrams** -- fenced code blocks with ` ```mermaid `
- **Task lists**, **footnotes**, **definition lists**, and **emoji**

See the [MkDocs Material reference](https://squidfunk.github.io/mkdocs-material/reference/) for full syntax.

### API Reference

API reference pages are auto-generated from Python docstrings. The `mkdocstrings` and `gen-files` plugins run `docs/gen_ref_pages.py` at build time to produce pages under `reference/`. You do not need to edit these files manually -- just write Google-style docstrings in `src/nemo_safe_synthesizer/` and they will appear on the next build.

### Deployment

Documentation is deployed to GitHub Pages automatically when changes to `docs/`, `mkdocs.yml`, or `src/` are pushed to `main`. The workflow is defined in `.github/workflows/docs.yml`.

## AI Agents

This project supports AI coding assistants (Cursor, Windsurf, Claude Code). Key files:
- **AGENTS.md** -- primary agent guide
- **.agent/skills/** -- domain-specific skills (canonical location)
- **.cursor/rules/** -- Cursor workflow rules
- **.cursor/skills/** -- symlinks to `.agent/skills/` for Cursor discoverability

Before contributing, run `make format` and `make lint`. See AGENTS.md for full conventions.

---

Thank you for contributing to NeMo Safe Synthesizer!
# Contributing to NeMo Safe Synthesizer

Thank you for your interest in contributing to NeMo Safe Synthesizer! This document provides guidelines and information for contributors.

Please read our [Code of Conduct](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.11+
- [mise](https://mise.jdx.dev/) -- Dev tool manager
- Git
- [gh](https://cli.github.com/) -- GitHub CLI (optional, for PR workflows)

!!! note
    Tools like [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), and [ty](https://github.com/astral-sh/ty) are installed automatically by `make bootstrap-tools` (via mise).

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

## Repository Settings

This repository uses GitHub Rulesets to enforce consistent contribution standards. These rules are automatically enforced -- you don't need to configure anything, but you should understand them to contribute successfully.

### Branch Naming Convention

All branches (except `main`) must follow this naming pattern:

```text
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

### Conventional Commits

All commits merged to `main` must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<scope>): <description>
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

!!! tip
    Since we use squash merging, your PR title should follow this format as it becomes the commit message.

### Branch Protection

The `main` branch has the following protections:

| Rule | Setting |
|------|---------|
| Required approvals | 1 |
| Code owner review | Required |
| Linear history | Required |
| Force pushes | Blocked |
| Merge strategy | Squash only |

## Pull Request Process

1. Create an issue first (if one doesn't exist) to discuss the change
2. Create a branch following the naming convention
3. Make your changes and commit using conventional commits
4. Run tests locally: `make test`
5. Push your branch and open a Pull Request
6. Address review feedback -- reviewers from CODEOWNERS will be automatically assigned
7. Once approved, your PR will be squash-merged and the branch auto-deleted

## Testing

```bash
# Run unit tests (excludes slow and e2e)
make test

# Run all tests including slow tests (excludes e2e)
make test-slow

# Run SDK-related tests
make test-sdk-related

# Run CI tests locally in a Linux container
make test-ci-container
```

## Code Style

### Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for code formatting and import sorting:

```bash
make format
```

### Linting and Type Checking

```bash
make lint
```

### Pre-commit Hooks

```bash
prek install
```

This installs hooks that run Ruff (format + lint), ty type checking, and uv lock verification on each commit.

## Developer Certificate of Origin

All contributions must be signed off:

```bash
git commit -s -m "feat: add new feature"
```

See the full [DCO](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/DCO) file for details.

---

Thank you for contributing to NeMo Safe Synthesizer!

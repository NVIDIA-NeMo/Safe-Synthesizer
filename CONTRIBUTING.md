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

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) - Python package manager
- [gh](https://cli.github.com/) - GitHub CLI (for running setup scripts)
- Git

### Setup

1. Fork the repository on GitHub

2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/safe-synthesizer.git
   cd safe-synthesizer
   ```

3. Set up the development environment:

   ```bash
   make bootstrap-python
   ```

4. Add the upstream remote:

   ```bash
   git remote add upstream https://github.com/NVIDIA-NeMo/safe-synthesizer.git
   ```

## Repository Settings

This repository uses GitHub Rulesets to enforce consistent contribution standards. These rules are automatically enforced—you don't need to configure anything, but you should understand them to contribute successfully.

### Branch Naming Convention

All branches (except `main`) must follow this naming pattern:

```
<author>/<issue-id>-<description>
```

or with an optional type:

```
<author>/<type>/<issue-id>-<description>
```

Rules:

- `<author>`: Your GitHub username (lowercase, alphanumeric, hyphens allowed)
- `<issue-id>`: The GitHub issue number (required)
- `<description>`: Brief description (lowercase, alphanumeric, hyphens)
- `<type>`: Optional category prefix

Valid types: `feature`, `bugfix`, `hotfix`, `release`, `docs`, `chore`, `test`

Examples:

| Branch Name | Valid |
|-------------|-------|
| `jsmith/123-add-login-feature` | ✅ |
| `jsmith/feature/123-add-login` | ✅ |
| `aagonzales/bugfix/456-fix-crash` | ✅ |
| `dev-team/docs/789-update-readme` | ✅ |
| `feature/add-login` | ❌ Missing author |
| `jsmith/add-login` | ❌ Missing issue ID |
| `JSmith/123-Add-Login` | ❌ Must be lowercase |

### Conventional Commits

All commits merged to `main` must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>
```

or without scope:

```
<type>: <description>
```

Rules:

- `<type>`: Required, must be one of the valid types below
- `<scope>`: Optional, indicates the area of the codebase affected
- `<description>`: Required, brief description (max 100 characters)
- Add `!` after type/scope for breaking changes

Valid types:

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, no logic change) |
| `refactor` | Code refactoring (no feature or fix) |
| `perf` | Performance improvements |
| `test` | Adding or updating tests |
| `build` | Build system or dependencies |
| `ci` | CI/CD configuration |
| `chore` | Maintenance tasks |
| `revert` | Reverting previous commits |

Examples:

| Commit Message | Valid |
|----------------|-------|
| `feat: add user authentication` | ✅ |
| `fix(auth): resolve token expiration bug` | ✅ |
| `docs: update API documentation` | ✅ |
| `chore(deps)!: bump major dependencies` | ✅ Breaking change |
| `Added new feature` | ❌ Missing type |
| `fix - resolve bug` | ❌ Wrong format |
| `FIX: resolve bug` | ❌ Type must be lowercase |

> **Note:** Since we use squash merging, your PR title should follow this format as it becomes the commit message.

### Semantic Versioning for Tags

Release tags must follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH[-prerelease][+build]
```

Examples:

| Tag | Valid |
|-----|-------|
| `1.0.0` | ✅ |
| `2.1.3` | ✅ |
| `1.0.0-alpha` | ✅ |
| `1.0.0-beta.1` | ✅ |
| `1.0.0-rc.1+build.123` | ✅ |
| `v1.0.0` | ❌ No `v` prefix |
| `release-1.0` | ❌ Wrong format |

### Branch Protection

The `main` branch has the following protections:

| Rule | Setting |
|------|---------|
| Required approvals | 1 |
| Code owner review | Required |
| Dismiss stale reviews | No |
| Require conversation resolution | Yes |
| Linear history | Required |
| Force pushes | Blocked |
| Deletions | Blocked |
| Merge strategy | Squash only |

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

- All files: `@NVIDIA-NeMo/safe-synthesizer-reviewers`
- Critical files (`pyproject.toml`, `uv.lock`, `SECURITY.md`, `LICENSE`, `.github/`): `@NVIDIA-NeMo/safe-synthesizer-maintainers`

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

```
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
# Run all unit tests
make test

# Run end-to-end tests (requires GPU)
make test-e2e

# Run specific test files
pytest tests/cli/test_run.py
```

### Test Requirements

Before submitting a PR:

- [ ] All existing tests pass
- [ ] New features include tests
- [ ] Bug fixes include regression tests

## Code Style

### Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
make lint

# Auto-fix issues
make lint-fix
```

### Type Checking

We use type hints throughout the codebase and use Ty for checking.


### Pre-commit Hooks

Consider setting up pre-commit hooks to catch issues early:

```bash
pre-commit install
```

---

Thank you for contributing to NeMo Safe Synthesizer!

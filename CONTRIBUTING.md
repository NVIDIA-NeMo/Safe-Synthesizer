# Contributing to NeMo Safe Synthesizer

Thank you for your interest in contributing to NeMo Safe Synthesizer! This document provides guidelines and information for contributors.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Table of Contents

- [Getting Started](#getting-started)
  - [Commit Signing](#commit-signing)
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
- Git 2.34+ (minimum required for SSH commit signing)

> Note: Other tools like [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [ty](https://github.com/astral-sh/ty), and [gh](https://cli.github.com/) are installed automatically by `make bootstrap-tools`.

### Setup

1. Get the code:

> NVIDIA employees have write access and can clone the repo directly. External contributors should fork first, then clone the fork and add an upstream remote.

  ```bash
   # NVIDIA internal -- clone directly
   git clone https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git

   # External -- fork on GitHub, then:
   git clone https://github.com/<your-username>/Safe-Synthesizer.git
   cd Safe-Synthesizer
   git remote add upstream https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git
  ```

2. Set up the development environment:

  ```bash
   cd Safe-Synthesizer

   # Install development tools (uv, ruff, ty, yq, etc.) to ~/.local/bin
   make bootstrap-tools

   # Ensure ~/.local/bin is on your PATH (add to your shell profile if needed)
   export PATH="$HOME/.local/bin:$PATH"

   # Install Python dependencies (choose one)
   make bootstrap-nss cpu    # CPU-only (macOS or Linux without GPU)
   make bootstrap-nss cuda   # CUDA 12.8 (Linux with NVIDIA GPU)
   make bootstrap-nss engine # Engine dependencies only
   make bootstrap-nss dev    # Minimal dev dependencies only
  ```

3. (Optional) Set a worktree base directory for working on multiple branches simultaneously. Add it to `.local.envrc` (git-ignored, auto-loaded by `.envrc`):

  ```bash
   echo 'export SS_WORKTREE_DIR="/path/to/worktrees"' >> .local.envrc
  ```

   Defaults to the parent of the repo root if unset. This is also useful for AI agents that create worktrees for isolated branch work. See the `git-worktrees` skill for details.

### Commit Signing

This repository requires [verified commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification). The `main` branch Ruleset enforces `required_signatures`, so unsigned commits will block PR merges. This is separate from [DCO sign-off](#developer-certificate-of-origin) -- both are required.

Choose one of the two options below.

#### Option A: SSH signing (recommended)

Most contributors already have an SSH key for GitHub authentication. The same key can also sign commits. If you don't have an SSH key yet, see [Generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

1. Set scopes on your `gh` cli. We'll remove them later.

   ```bash
   gh auth refresh -s admin:ssh_signing_key
   ```

2. Check whether your key is already registered for signing:

   ```bash
   gh ssh-key list
   ```

   If your key already appears with type `signing`, skip to step 3.

3. Register the key as a signing key on GitHub (authentication and signing keys are tracked separately -- having one does not count as the other). This registers the key and then removes the permission scope so it doesn't persist in your token (change this if you want to keep the scope).

   ```bash
     gh ssh-key add ~/.ssh/id_ed25519.pub --type signing \
     && gh auth refresh -r admin:ssh_signing_key
   ```

   Or [manually via GitHub Settings](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-new-ssh-key-to-your-github-account) > SSH and GPG keys > New SSH key > Key type: "Signing Key".

4. Configure git to sign commits (see [Telling Git about your signing key](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key) for details):

!!! info "git global"
    You can make this a global default if you'd like by adding the `--global` flag. The following commands are repo scoped.

   ```bash
   git config gpg.format ssh
   git config user.signingkey ~/.ssh/id_ed25519.pub
   ```

5. (Optional) Configure local verification:

   To see "Good signature" locally when running `git log --show-signature`, git needs to know which SSH keys to trust.

   ```bash
   # Create allowed_signers file
   echo "$(git config --get user.email) $(cat ~/.ssh/id_ed25519.pub)" >> ~/.ssh/allowed_signers

   # Tell git to use it
   git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers
   ```

#### Option B: GPG signing

If you already have a GPG key or prefer GPG. To generate one, see [Generating a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

1. Register the key on GitHub. The `admin:gpg_key` scope grants write access to your account's GPG keys; the one-liner below adds it, uploads the key, then removes the scope:

   ```bash
   gh auth refresh -s admin:gpg_key \
     && gh gpg-key add <public-key-file> \
     && gh auth refresh -r admin:gpg_key
   ```

   Or [manually via GitHub Settings](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-gpg-key-to-your-github-account) > SSH and GPG keys > New GPG key.

2. Configure git to use your key to sign commits:

   ```bash
   git config user.signingkey <GPG-KEY-ID>
   ```

#### Verify signing works

```bash
git commit --allow-empty -s -S -m "test: verify commit signing"
git log --show-signature -1

# Clean up the test commit
git reset --soft HEAD~1
```

You should see a valid signature in the output. On GitHub, the commit will display a "Verified" badge. If something isn't working, see [Troubleshooting commit signature verification](https://docs.github.com/en/authentication/troubleshooting-commit-signature-verification).


To avoid forgetting `--signoff` and `--gpg-sign` on future commits, configure this repo to GPG-sign automatically and create a short alias that adds DCO sign-off:


!!! info "Git aliases"
    You can obviously choose your own aliases or set them elsewhere - this is just a suggestion so you do not have to think about it.


```bash
# Automatic GPG signing on every commit (native git config)
git config commit.gpgsign true

# Alias -- git aliases can't override built-in commands, so use "commit-sign" instead of "commit"
git config alias.commit-sign "commit --signoff"
```

Then use `git commit-sign` instead of `git commit`. Since `commit.gpgsign` is active, every commit is both signed and DCO-certified.

NVIDIA internal contributors who work primarily on repos that require DCO and signing can set these globally instead: `git config --global commit.gpgsign true` and `git config --global alias.commit-sign "commit --signoff"`.

#### Re-signing existing commits

If you have unsigned commits on a feature branch that were pushed before signing was configured, rebase to re-create them with signatures. Use the remote that points to the NVIDIA repo (`origin` for internal contributors, `upstream` for external forks):

```bash
# NVIDIA internal
git rebase --force-rebase --gpg-sign --signoff origin/main

# External (forked)
git rebase --force-rebase --gpg-sign --signoff upstream/main

git push --force-with-lease
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


| Rule                            | Setting      |
| ------------------------------- | ------------ |
| Required approvals              | 1            |
| Code owner review               | Required     |
| Dismiss stale reviews           | Yes          |
| Require conversation resolution | Yes          |
| Signed commits                  | Required     |
| Required status checks          | CI Status    |
| Linear history                  | Required     |
| Force pushes                    | Blocked      |
| Deletions                       | Blocked      |
| Merge strategy                  | Squash only  |


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
   - Respond to comments in the github console, be sure to submit as pending comments are only visible to you
   - Resolve comments where the requested change has been made or otherwise addressed.
   - Leave comments unresolved if seeking further review or input from the reviewer
   - Reviewers may re-open resolved comments with further comments or questions, that's okay and part of the process
   - After responding to all comments and pushing changes to the branch, re-request review with the circular arrow button to the right of the reviewer name
   - Use the Assignees list to indicate who's expected to take the next action on the PR, such as PR author after reviewer leaves comments, or the reviewer after updates have been made
   - Reviewers: If there is an error in the PR or something that requires large changes, review and mark it as "requires changes" for explicit feedback. This can give signal for triaging which PRs are mostly ready or those that require more work.
8. Merge — once approved, your PR will be squash-merged and the branch auto-deleted. Please review the git message, which will automatically be set to the first comment in the PR.

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

> Note: DCO sign-off (`git commit -s`) adds a text trailer asserting your right to contribute. It is not a cryptographic signature. This repository also requires [commit signing](#commit-signing) -- both are independent requirements.

## Testing

### Running Tests

```bash
# Run unit tests (excludes slow and e2e)
make test

# Run all tests including slow tests (excludes e2e)
make test-slow

# Run GPU integration tests (requires CUDA)
make test-gpu-integration

# Run end-to-end tests (requires CUDA)
make test-e2e

# Run a specific config-dataset e2e combo (12 total, see tests/TESTING.md)
make test-nss-tinyllama_unsloth-clinc_oos-ci

# Run CI tests locally in a Linux container (Docker/Podman)
make test-ci-container

# Run specific test files directly
uv run pytest tests/cli/test_run.py
```

### GPU E2E Tests (CI)

GPU E2E tests run on NVIDIA self-hosted A100 runners and require the copy-pr-bot setup -- they cannot run on a local machine unless you have a compatible GPU environment.

When you open a ready-for-review PR, copy-pr-bot automatically triggers a GPU test run. For draft PRs, or to re-run after a flaky failure, comment `/sync` on the PR. The bot will push the current HEAD to `pull-request/<number>`, fire `gpu-tests.yml`, and post the `GPU CI Status` check result back to the PR.

To trigger from the CLI instead (no PR status check):

```bash
gh workflow run gpu-tests.yml --ref <your-branch>
```

### Test Requirements

Before submitting a PR:

- All existing tests pass (`make test`)
- New features include tests
- Bug fixes include regression tests

## Code Style

For detailed style guidelines covering Python, markdown, Dockerfiles, shell scripts, testing, and docstrings, see [STYLE_GUIDE.md](STYLE_GUIDE.md).

### Formatting, Linting, and Type Checking

Use `make` targets instead of running `ruff` or `ty` directly. The targets use pinned tool versions from `make bootstrap-tools` and check all tracked files.

```bash
make format   # auto-fix: ruff format + import sorting + copyright headers
make check    # read-only: all CI checks (format + lint + typecheck + copyright)
make test     # unit tests
# or just
make format check test
```

We use `ruff` and `ty` for the majority of this work, wrapped with settings for consistency.

CI calls the same tools through atomic read-only `make` targets, so the Makefile is the single source of truth for how each check runs. `make check` replicates all CI code-quality checks locally (format-check + typecheck). Pre-commit hooks (`pre-commit install`) provide faster feedback by checking only staged files, but are not a substitute for the `make` targets.

The wrapper scripts in `tools/` also accept explicit file paths for spot-checking individual files:

```bash
bash tools/codestyle/format.sh --check src/nemo_safe_synthesizer/cli/run.py
bash tools/codestyle/ruff_check.sh src/nemo_safe_synthesizer/cli/run.py
```

All source files (`.py`, `.sh`, `.yaml`, `.yml`, `.md`) require SPDX copyright headers. `make format` adds them automatically; exclusions are listed in `.copyrightignore`.

All `make` targets check the entire project. Pre-commit scopes checks to staged files. The wrapper scripts also accept explicit file paths when you want to check specific files.

| Check | CI target | `make format` / `make check` | Pre-commit |
|---|---|---|---|
| ruff format + lint | `make format-check` | `format`: auto-fix; `check`: read-only | staged files (auto-fix) |
| ty typecheck | `make typecheck` | read-only | all files |
| copyright headers | `make format-check` | `format`: auto-fix; `check`: read-only | staged files (auto-fix) |
| uv lock drift | `make lock-check` | not checked | on `pyproject.toml` changes |
| DCO signoff | branch protection | not checked | commit-msg hook |

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

In Cursor or VS Code Remote, the port is auto-forwarded. Check the Ports
panel (`Ctrl+Shift+P` > "Ports: Focus on Ports View") -- port 8000 will
appear with a local address you can open in the Simple Browser or your
system browser.

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

- Admonitions -- callout boxes (`!!! note`, `!!! warning`, `??? tip` for collapsible)
- Content tabs -- tabbed content blocks (`=== "Python SDK"` / `=== "CLI"`)
- Code blocks -- syntax highlighting, line numbers, copy button, and annotations
- Mermaid diagrams -- fenced code blocks with ` ```mermaid `
- Task lists, footnotes, definition lists, and emoji

See the [MkDocs Material reference](https://squidfunk.github.io/mkdocs-material/reference/) for full syntax.

### API Reference

API reference pages are auto-generated from Python docstrings. The `mkdocstrings` and `gen-files` plugins run `docs/gen_ref_pages.py` at build time to produce pages under `reference/`. You do not need to edit these files manually -- just write Google-style docstrings in `src/nemo_safe_synthesizer/` and they will appear on the next build.

### Deployment

Documentation is deployed to GitHub Pages automatically when changes to `docs/`, `mkdocs.yml`, or `src/` are pushed to `main`. The workflow is defined in `.github/workflows/docs.yml`.

## AI Agents

This project supports AI coding assistants. Configuration is layered so that conventions are shared across tools while tool-specific features use their native config format.

| Config file | Read by | Purpose |
|-------------|---------|---------|
| `AGENTS.md` | All agents (Cursor, Windsurf, Claude Code, etc.) | Repo conventions, module map, skills index |
| `AGENTS.local.md` | All agents | Local developer preferences (git-ignored) |
| `CLAUDE.md` | Claude Code | Entry point; references `AGENTS.md` and `AGENTS.local.md` |
| `.cursor/rules/*.mdc` | Cursor only | Workflow rules, style enforcement, file-pattern triggers |
| `.agent/skills/*/SKILL.md` | All agents (via skills index in `AGENTS.md`) | Domain-specific knowledge (testing, sync, typing, etc.) |
| `.cursor/skills/` | Cursor only | Symlinks to `.agent/skills/` for Cursor discoverability |
| `src/**/AGENTS.md`, `tests/AGENTS.md` | All agents | Per-module guides for non-obvious patterns and gotchas |

Conventions defined in `AGENTS.md` (code style, markdown style, testing, etc.) apply universally. Tool-specific config (`.cursor/rules/`, `CLAUDE.md`) reinforces those conventions for its respective tool.

Before contributing, run `make format` and `make check`. See `AGENTS.md` for full conventions.

---

Thank you for contributing to NeMo Safe Synthesizer!

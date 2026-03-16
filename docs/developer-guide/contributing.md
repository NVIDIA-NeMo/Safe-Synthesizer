<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing

Thank you for your interest in contributing to NeMo Safe Synthesizer. This page covers setup, repository rules, the pull request process, testing, code style, and documentation. Please read the [Code of Conduct](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/CODE_OF_CONDUCT.md) before contributing.

!!! note "Sync with CONTRIBUTING.md"
    `CONTRIBUTING.md` is the canonical source for contribution process policy. Keep this page synchronized so GitHub and docs guidance stay aligned.

## Getting Started

### Prerequisites

- Python 3.11+
- Git 2.34+ (minimum required for SSH commit signing)

Other tools ([uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [ty](https://github.com/astral-sh/ty), [gh](https://cli.github.com/)) are installed by `make bootstrap-tools`.

### Setup

1. Get the code:

=== "NVIDIA internal"

    Clone directly (NVIDIA employees have write access):

    ```bash
    git clone https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git
    ```

=== "External (fork)"

    Fork on GitHub, then clone your fork and add the upstream remote:

    ```bash
    git clone https://github.com/<your-username>/Safe-Synthesizer.git
    cd Safe-Synthesizer
    git remote add upstream https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git
    ```

2. Set up the development environment:

If you are looking for end-user installation paths instead of contributor setup, see [User Guide -- Getting Started](../user-guide/getting-started.md).

```bash
cd Safe-Synthesizer

# Install development tools (uv, ruff, ty, yq, etc.) to ~/.local/bin
make bootstrap-tools

# Ensure ~/.local/bin is on your PATH (add to your shell profile if needed)
export PATH="$HOME/.local/bin:$PATH"

# Install Python dependencies (choose one)
make bootstrap-nss cpu    # CPU-only (macOS or Linux without GPU)
make bootstrap-nss cuda   # CUDA 12.8 (Linux with NVIDIA GPU; maps to the `cu128` extra)
make bootstrap-nss engine # Engine dependencies only
make bootstrap-nss dev    # Minimal dev dependencies only
```

3. (Optional) Set a worktree base directory for working on multiple branches. Add to `.local.envrc` (git-ignored, auto-loaded by `.envrc`):

```bash
echo 'export SS_WORKTREE_DIR="/path/to/worktrees"' >> .local.envrc
```

Defaults to the parent of the repo root if unset. See the `git-worktrees` skill for details.

### Commit Signing

This repository requires [verified commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification). The `main` branch Ruleset enforces `required_signatures`; unsigned commits block PR merges. This is separate from [DCO sign-off](#developer-certificate-of-origin) — both are required.

=== "SSH (recommended)"

    Most contributors already have an SSH key for GitHub; the same key can sign commits. If you need a key, see [Generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

    1. Set scopes on `gh` (we'll remove them later):

    ```bash
    gh auth refresh -s admin:ssh_signing_key
    ```

    2. Check whether your key is already registered for signing:

    ```bash
    gh ssh-key list
    ```

    If your key appears with type `signing`, skip to step 3.

    3. Register the key as a signing key on GitHub:

    ```bash
    gh ssh-key add ~/.ssh/id_ed25519.pub --type signing \
      && gh auth refresh -r admin:ssh_signing_key
    ```

    Or [manually via GitHub Settings](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-new-ssh-key-to-your-github-account) > SSH and GPG keys > New SSH key > Key type: "Signing Key".

    4. Configure git to sign commits:

    !!! info "git global"
        Add the `--global` flag to make this your default. The following are repo-scoped.

    ```bash
    git config gpg.format ssh
    git config user.signingkey ~/.ssh/id_ed25519.pub
    ```

    5. (Optional) Local verification — so `git log --show-signature` shows "Good signature":

    ```bash
    echo "$(git config --get user.email) $(cat ~/.ssh/id_ed25519.pub)" >> ~/.ssh/allowed_signers
    git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers
    ```

=== "GPG"

    If you use a GPG key, see [Generating a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

    1. Register the key on GitHub:

    ```bash
    gh auth refresh -s admin:gpg_key \
      && gh gpg-key add <public-key-file> \
      && gh auth refresh -r admin:gpg_key
    ```

    Or [manually via GitHub Settings](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-gpg-key-to-your-github-account).

    2. Configure git:

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

You should see a valid signature; on GitHub the commit shows a "Verified" badge. See [Troubleshooting commit signature verification](https://docs.github.com/en/authentication/troubleshooting-commit-signature-verification) if needed.

To avoid forgetting `--signoff` and signing, enable automatic signing and an alias:

!!! info "Git aliases"
    You can choose your own aliases or set them elsewhere.

```bash
git config commit.gpgsign true
git config alias.commit-sign "commit --signoff"
```

Use `git commit-sign` instead of `git commit` so every commit is signed and DCO-certified. NVIDIA internal contributors can set these globally: `git config --global commit.gpgsign true` and `git config --global alias.commit-sign "commit --signoff"`.

#### Re-signing existing commits

If you have unsigned commits on a feature branch, rebase to re-create them with signatures:

```bash
# NVIDIA internal
git rebase --force-rebase --gpg-sign --signoff origin/main

# External (forked)
git rebase --force-rebase --gpg-sign --signoff upstream/main

git push --force-with-lease
```

### NMP Integration

NeMo Safe Synthesizer is a standalone package. Changes flow into NMP via Artifactory publishing and vendor packaging. See the [NMP Integration](https://github.com/NVIDIA-NeMo/Safe-Synthesizer#nmp-integration) section of the README for publishing, SDK vendoring, and local development workflows.

## Repository Settings

This repository uses GitHub Rulesets to enforce contribution standards. You don't need to configure anything, but you should understand the rules.

??? note "Repository Settings (branch naming, commits, tags, protection)"

    ### Branch Naming Convention

    All branches except `main` must follow:

    ```text
    <author>/<description>
    <author>/<issue-id>-<description>
    <author>/<type>/<description>
    <author>/<type>/<issue-id>-<description>
    ```

    - `<author>`: GitHub username (lowercase, alphanumeric, hyphens)
    - `<issue-id>`: Optional issue number prefix (e.g. `123-`)
    - `<description>`: Brief description (lowercase, alphanumeric, hyphens)
    - `<type>`: Optional — `feature`, `bugfix`, `hotfix`, `release`, `docs`, `chore`, `test`

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

    All commits merged to `main` must follow [Conventional Commits](https://www.conventionalcommits.org/):

    ```text
    <type>(<scope>): <description>
    ```

    or without scope: `<type>: <description>`. Add `!` after type/scope for breaking changes.

    Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

    | Commit Message                            | Valid             |
    | ----------------------------------------- | ----------------- |
    | `feat: add user authentication`           | ✅                 |
    | `fix(auth): resolve token expiration bug` | ✅                 |
    | `docs: update API documentation`          | ✅                 |
    | `chore(deps)!: bump major dependencies`   | ✅ Breaking change |
    | `Added new feature`                       | ❌ Missing type   |
    | `FIX: resolve bug`                        | ❌ Type lowercase |

    We use squash merging; the PR title becomes the commit message.

    ### Semantic Versioning for Tags

    Release tags must follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH[-prerelease][+build]`. No `v` prefix.

    ### Branch Protection

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

1. Create an issue first (if one doesn't exist) to discuss the change.
2. Create a branch following the [naming convention](#repository-settings): `git checkout -b <username>/<issue-id>-<description>`.
3. Make your changes and commit using [conventional commits](#repository-settings).
4. Complete the pre-review checklist before requesting review:
    - [ ] `make format && make check` passes (or via pre-commit validation)
    - [ ] `make test` passes locally
    - [ ] `make test-e2e` passes locally (requires CUDA)
    - [ ] `make test-ci` passes locally (recommended)
    - [ ] `make test-ci-slow` passes locally when touching slow-test areas
    - [ ] `make test-ci-container` passes locally (recommended)
5. Push: `git push origin <your-branch>`.
6. Open a Pull Request using the [PR template](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/.github/PULL_REQUEST_TEMPLATE.md). You can submit a draft early to signal the issue is being worked on.
7. Address review feedback — [CODEOWNERS](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/.github/CODEOWNERS) assign reviewers automatically:
    - Respond to comments in the GitHub UI. Pending comments are only visible to you until submitted.
    - Resolve a comment when the requested change has been made or otherwise addressed. Leave it unresolved if you're seeking further input from the reviewer.
    - Reviewers may re-open resolved comments with follow-up questions -- that's normal.
    - After pushing updates, re-request review using the circular arrow button next to the reviewer's name.
    - Use the Assignees field to signal whose turn it is to act (author after reviewer comments, reviewer after updates).
    - Reviewers: mark a PR "Requires changes" when there are errors or significant rework needed -- this helps triage which PRs are nearly ready vs. need more work.
8. Before merge, ensure the pre-merge checklist is complete:
    - [ ] New or updated tests for any fix or new behavior
    - [ ] Updated documentation for new features and behaviors, including docstrings for API docs
9. Merge -- once approved, the PR is squash-merged and the branch is auto-deleted. The PR title becomes the commit message; review it before merging.

### CODEOWNERS

- `src/` and `tests/`: `@NVIDIA-NeMo/safe-synthesizer-reviewers`
- Remaining files (e.g. `pyproject.toml`, `uv.lock`, `SECURITY.md`, `.github/`): `@NVIDIA-NeMo/safe-synthesizer-maintainers`

## Issues and Discussions

- Issue templates: Bug Report, Feature Request, Development Task.
- For general questions use [GitHub Discussions](https://github.com/NVIDIA-NeMo/safe-synthesizer/discussions) instead of opening an issue.

## Developer Certificate of Origin

All contributions must be signed off: add a `Signed-off-by` line to commit messages.

```bash
git commit -s -m "feat: add new feature"
```

This adds a line like `Signed-off-by: Your Name <your.email@example.com>`. By signing off you certify the [Developer Certificate of Origin](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/DCO). DCO sign-off is a text trailer, not a cryptographic signature; this repo also requires [commit signing](#commit-signing) — both are required.

## Testing

Run unit tests before submitting a PR: `make test`. All existing tests must pass; new features need tests and bug fixes need regression tests.

??? tip "Test commands reference"

    | Command | Description |
    | -------- | ------------ |
    | `make test` | Unit tests (excludes slow and e2e) |
    | `make test-slow` | All tests including slow (excludes e2e) |
    | `make test-ci` | CI unit tests with coverage (excludes slow, e2e, gpu) |
    | `make test-ci-slow` | CI slow tests with coverage |
    | `make test-gpu-integration` | GPU integration tests (requires CUDA) |
    | `make test-e2e` | All end-to-end tests (requires CUDA) |
    | `make test-e2e-default` | Default e2e tests (requires CUDA) |
    | `make test-e2e-dp` | Differential privacy e2e tests (requires CUDA) |
    | `make test-e2e-collect` | Dry-run: show which tests e2e targets select |
    | `make test-nss-tinyllama_unsloth-clinc_oos-ci` | Specific e2e combo (see `tests/TESTING.md`) |
    | `make test-ci-container` | CI tests in a Linux container (Docker/Podman) |
    | `uv run pytest tests/cli/test_run.py` | Run specific test files |

## Code Style

See [STYLE_GUIDE.md](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/STYLE_GUIDE.md) for Python, markdown, Dockerfiles, shell scripts, testing, and docstrings.

Use `make` targets instead of running `ruff` or `ty` directly. The targets use pinned tool versions from `make bootstrap-tools` and check all tracked files:

```bash
make format         # auto-fix: ruff format + import sorting + copyright headers
make check          # read-only: all CI checks (format + lint + typecheck + copyright)
make test           # unit tests
make format check test  # chain them
```

`make check` replicates all CI code-quality checks locally. Pre-commit hooks (`pre-commit install`) provide faster feedback on staged files but are not a substitute for the `make` targets.

The `tools/` wrapper scripts also accept explicit file paths for spot-checking individual files:

```bash
bash tools/codestyle/format.sh --check src/nemo_safe_synthesizer/cli/run.py
bash tools/codestyle/ruff_check.sh src/nemo_safe_synthesizer/cli/run.py
```

All source files (`.py`, `.sh`, `.yaml`, `.yml`, `.md`) require SPDX copyright headers. `make format` adds them automatically; exclusions are listed in `.copyrightignore`.

| Check | CI target | `make format` / `make check` | Pre-commit |
| ----- | --------- | ---------------------------- | ---------- |
| ruff format + lint | `make format-check` | `format`: auto-fix; `check`: read-only | staged (auto-fix) |
| ty typecheck | `make typecheck` | read-only | all files |
| copyright headers | `make format-check` | auto-fix / read-only | staged (auto-fix) |
| uv lock drift | `make lock-check` | not checked | on `pyproject.toml` changes |
| DCO signoff | branch protection | not checked | commit-msg hook |

## Documentation

This site uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/), hosted at <https://nvidia-nemo.github.io/Safe-Synthesizer/>.

### Local Preview

Documentation dependencies are included in the `dev` bootstrap profile. If you ran `make bootstrap-nss dev` (or `cpu`/`cuda`), you're already set. Otherwise install them directly:

```bash
uv sync --group docs
```

Start a local server with live reload:

```bash
make docs-serve
# Browse to http://127.0.0.1:8000
```

In Cursor or VS Code Remote, the port is auto-forwarded. Check the Ports panel (`Ctrl+Shift+P` > "Ports: Focus on Ports View") -- port 8000 will appear with a local address you can open in the Simple Browser or your system browser.

Build the static site (output in `site/`):

```bash
make docs-build
```

### Directory Layout

All documentation lives under `docs/`. The structure follows the [Diataxis](https://diataxis.fr/) framework:

| Directory | Content type | Examples |
| --- | --- | --- |
| `tutorials/` | Tutorials | Hands-on walkthroughs |
| `user-guide/` | How-tos & reference | CLI, configuration, SDK |
| `developer-guide/` | Explanations | Contributing, architecture |
| `product-overview/` | Explanations | Pipeline and feature overviews |
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

API reference pages are auto-generated from Python docstrings. The `mkdocstrings` and `gen-files` plugins run `docs/gen_ref_pages.py` at build time to produce pages under `reference/`. Write Google-style docstrings in `src/nemo_safe_synthesizer/` and they will appear on the next build -- no manual edits to `reference/` needed.

### Deployment

Documentation is deployed to GitHub Pages automatically when changes to `docs/`, `mkdocs.yml`, or `src/` are pushed to `main`. The workflow is defined in `.github/workflows/docs.yml`.

## AI Agents

This project supports AI coding assistants. Configuration is layered so that conventions are shared across tools while tool-specific features use their native config format:

| Config file | Read by | Purpose |
|---|---|---|
| `AGENTS.md` | All agents (Cursor, Windsurf, Claude Code, etc.) | Repo conventions, module map, skills index |
| `AGENTS.local.md` | All agents | Local developer preferences (git-ignored) |
| `CLAUDE.md` | Claude Code | Entry point; references `AGENTS.md` and `AGENTS.local.md` |
| `.cursor/rules/*.mdc` | Cursor only | Workflow rules, style enforcement, file-pattern triggers |
| `.agent/skills/*/SKILL.md` | All agents (via skills index in `AGENTS.md`) | Domain-specific knowledge (testing, sync, typing, etc.) |
| `.cursor/skills/` | Cursor only | Symlinks to `.agent/skills/` for Cursor discoverability |
| `src/**/AGENTS.md`, `tests/AGENTS.md` | All agents | Per-module guides for non-obvious patterns and gotchas |

Conventions defined in `AGENTS.md` (code style, markdown style, testing, etc.) apply universally. Tool-specific config (`.cursor/rules/`, `CLAUDE.md`) reinforces those conventions for its respective tool.

Before contributing, run `make format` and `make check`. See `AGENTS.md` for full conventions.

## See also

<div class="grid cards" markdown>

-   Architecture

    ---

    High-level design, components, execution flow, and extension points.

    [:octicons-arrow-right-24: Architecture](architecture.md)

-   API Reference

    ---

    Auto-generated API docs from Python docstrings.

    [:octicons-arrow-right-24: API Reference](../reference/)

-   Code of Conduct

    ---

    Community standards and expectations for contributors.

    [:octicons-arrow-right-24: Code of Conduct](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/CODE_OF_CONDUCT.md)

</div>

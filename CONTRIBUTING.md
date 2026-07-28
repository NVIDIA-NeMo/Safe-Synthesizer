# Contributing to NeMo Safe Synthesizer

Thank you for your interest in contributing to NeMo Safe Synthesizer! This document provides guidelines and information for contributors.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Table of Contents

- [Getting Started](#getting-started)
  - [Commit Signing](#commit-signing)
- [Repository Settings](#repository-settings)
  - [Branch Naming Convention](#branch-naming-convention)
  - [Conventional Commits](#conventional-commits)
  - [Branch Protection](#branch-protection)
- [Classification and Project Tracking](#classification-and-project-tracking)
- [Pull Request Process](#pull-request-process)
- [Issues and Discussions](#issues-and-discussions)
- [Developer Certificate of Origin](#developer-certificate-of-origin)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [AI Agents](#ai-agents)
- [Releasing](#releasing)
- [NMP Integration](#nmp-integration)

## Getting Started

### Prerequisites

- Python 3.11–3.13 (project supports Python 3.11, 3.12, and 3.13; `.python-version` pins 3.13 for bootstrapping at the repo root. Python 3.14+ is not supported — see [Troubleshooting](docs/user-guide/troubleshooting.md#python-314-is-not-supported))
- Git 2.34+ (minimum required for SSH commit signing)

> Note: Other tools like [uv](https://docs.astral.sh/uv/), [dprint](https://dprint.dev/), [ruff](https://docs.astral.sh/ruff/), [ty](https://github.com/astral-sh/ty), and [gh](https://cli.github.com/) are installed automatically by `make setup` (via [mise](https://mise.jdx.dev/)). Tool versions are declared in `.mise.toml` and locked in `mise.lock` (committed), ensuring reproducible toolchains across developer systems and CI. These should not interfere with locally installed tools.

> Note on mise itself: the mise version is pinned in `.mise.toml` (`min_version`). The first run of `make setup` installs exactly that version via `tools/install-mise.sh`, preferring the GPG-verified installer when the full toolchain (`gpg`, `gpg-agent`, and `dirmngr`) is available and falling back to `https://mise.run` otherwise (with a warning). If you already have a different mise version on `PATH`, `make setup` will stop and tell you -- either run `mise self-update <pinned>` or uninstall the existing mise and rerun. It will not silently replace your install.

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

   # Install dev tools via mise (installs mise itself if missing)
   make setup

   # Install Python dependencies (choose one)
   mise run bootstrap-nss cpu    # CPU-only (macOS or Linux without GPU)
   mise run bootstrap-nss cuda   # CUDA 12.9 (Linux with NVIDIA GPU)
   mise run bootstrap-nss engine # Engine dependencies only
   mise run bootstrap-nss dev    # Minimal dev dependencies only
  ```

3. (Optional) If you use git worktrees or AI agents that create worktrees, add mise trust for worktree paths. Without this, tools and env vars won't load in new worktree directories:

  ```bash
   # Add to your shell profile (~/.bashrc, ~/.zshrc)
   REPO="$(cd "$(git rev-parse --show-toplevel)" && pwd -P)"
   printf 'export MISE_TRUSTED_CONFIG_PATHS="%s"\n' "$REPO" >> ~/.bashrc   # or ~/.zshrc
  ```

  Alternatively, set `MISE_YES=1` to trust all configs globally (appropriate for dev machines and CI).

4. (Optional) Set a worktree base directory for working on multiple branches simultaneously. Add it to `.env.local` (git-ignored, auto-loaded by mise):

  ```bash
   # .env.local -- project-local overrides (git-ignored)
   SS_WORKTREE_DIR="/path/to/worktrees"
  ```

   Defaults to the parent of the repo root if unset. This is also useful for AI agents that create worktrees for isolated branch work. See the `git-worktrees` skill for details.

   mise automatically loads `.env` and `.env.local` from the project root (configured in `.mise.toml`). Use `.env` for shared defaults and `.env.local` for machine-specific overrides -- both are git-ignored.

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

   If your key already appears with type `signing`, skip to step 4.

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

`GPU CI Status` is not currently a live branch-protection requirement because PR GPU runs are blocked by internal issues. We expect to re-enable it as soon as those blockers are resolved.

## Classification and Project Tracking

Three mechanisms classify and track work in this repo:

1. GitHub issue Type field -- org-level, single value per issue.
2. Labels -- repo-level, multiple per issue or PR; the canonical list lives below.
3. Safe Synthesizer Development GitHub project fields -- `Priority` and `Size`.

The split is intentional: the Type field handles coarse classification, labels handle granularity the Type field can't express, and the project handles backlog ordering and effort tracking.

### Issue Type

The GitHub Type field is set at the NVIDIA-NeMo org level (changing the value list requires an org owner). It applies to issues only -- PRs use the type labels described below.

| Type | Use for |
| --- | --- |
| `Bug` | Defects in shipped behavior |
| `Feature` | New user-facing capability |
| `Question` | Questions or clarification requests; prefer [GitHub Discussions](https://github.com/NVIDIA-NeMo/safe-synthesizer/discussions) for general questions |
| `Task` | Internal development work, refactors, infrastructure |

Most internal issues will be `Task`.

### Project fields: Priority and Size

Issues are tracked in the Safe Synthesizer Development GitHub project. Priority and effort estimates live as project fields, not labels, so backlog and prioritization views can sort and group on them directly.

- `Priority` -- relative urgency for backlog ordering. 
- `Size` -- rough effort estimate.

Set both in the [project view](https://github.com/orgs/NVIDIA-NeMo/projects/74) or expand the Projects drop-down on the right pane for PRs and issues.

### Labels

Labels are repo-scoped and apply to both issues and PRs. There is no required minimum set per issue or PR; apply labels that add useful signal.

#### Type

Roughly aligned with our Conventional Commit types. Apply when it adds signal; multiple are fine for cross-cutting work.

| Label | Use for |
| --- | --- |
| `bug` | Defects in shipped behavior |
| `feature` | New user-facing capability |
| `refactor` | Internal restructuring with no behavior change |
| `perf` | Performance improvement |
| `security` | Security-relevant fix or hardening |
| `docs` | Documentation-only change |
| `test` | Test-only addition or change |
| `chore` | Maintenance not tied to a user-visible change |
| `breaking-change` | Backward-incompatible change (paired with another type) |

#### Area

Where in the codebase the work lands. Optional on issues.

On PRs, `area:*` labels are auto-applied based on the files changed, using the rules in [`.github/labeler.yml`](.github/labeler.yml). Labels are additive only -- maintainers can add, remove, or override them manually, and the action will not undo manual changes on subsequent pushes. Update the labeler rules in the same PR that introduces new modules or area labels.

Product areas mirror the `src/nemo_safe_synthesizer/` module map in [`AGENTS.md`](AGENTS.md):

`area:sdk-cli`, `area:config`, `area:data-processing`, `area:evaluation`, `area:generation`, `area:training`, `area:pii`, `area:privacy`, `area:llm`, `area:observability`.

Infrastructure areas:

`area:ci`, `area:tests`, `area:docs`, `area:dev-ex`, `area:build-dist`.

#### Merge status

| Label | Use for |
| --- | --- |
| `ready-to-merge` | All review and CI gates satisfied; ready for a maintainer to merge |
| `blocked` | Cannot progress until an external dependency or upstream change is resolved |

Other "who acts next" signaling on PRs is done via Assignees -- see the [Pull Request Process](#pull-request-process).

#### Workflow

| Label | Use for |
| --- | --- |
| `community-request` | Issue or PR from an external contributor or user |
| `stale` | Auto-applied to inactive issues/PRs |

#### Dependabot

Dependabot manages `dependencies` and `github_actions` automatically on the PRs it opens; don't apply them manually.

### Adding or changing labels

This section is the canonical source. To add, rename, or remove a label, open a PR that updates this section and apply the change in the GitHub UI in the same PR. If drift becomes a problem we can switch to a `.github/labels.yml` driven by a sync action.

## Pull Request Process

1. Create an issue first (if one doesn't exist) to discuss the change
2. Create a branch following the [naming convention](#branch-naming-convention):
  ```bash
   git checkout -b <username>/<issue-id>-<description>
  ```
3. Make your changes and commit using [conventional commits](#conventional-commits)
4. Run tests locally:
  ```bash
   mise run test
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

See [tests/TESTING.md](tests/TESTING.md) for the full test matrix and usage.

### Running Tests

```bash
# Run unit tests (excludes slow unit tests, smoke and e2e)
mise run test

# Run all unit tests including slow tests (excludes smoke and e2e)
mise run test:unit-slow

# Run CPU smoke tests (~few min, no GPU required)
mise run test:smoke

# Run GPU smoke tests (requires CUDA)
mise run test:smoke:gpu

# Run end-to-end tests (requires CUDA)
mise run test:e2e

# Run a specific config-dataset e2e combo (12 total, see tests/TESTING.md)
mise run test-nss-tinyllama_nodp-clinc_oos-ci

# Run CI tests locally in a Linux container (Docker/Podman)
mise run test:ci-container

# Run specific test files directly
uv run --frozen pytest tests/cli/test_run.py
```

`mise run test:e2e:default` and `mise run test:e2e:dp` each run the SafeSynthesizer e2e flow across Mistral 7B, SmolLM3 3B, and TinyLlama 1.1B. Each model case has a 30-minute timeout, so reserve up to 90 minutes for either target and up to 3 hours for the full `mise run test:e2e` target in cold-cache GPU environments.

### GPU Tests (CI)

GPU tests run on NVIDIA self-hosted A100 runners -- they cannot run on a local machine unless you have a compatible GPU environment. `gpu-tests.yml` currently runs only on the nightly schedule or manual `workflow_dispatch`; the `push` trigger for copy-pr-bot PR branches is commented out due to internal blockers. We expect to re-enable PR GPU runs as soon as those blockers are resolved. The workflow has two main test jobs:

- GPU Smoke Tests -- staged smoke tests (train-only, generation, resume, structured gen, timeseries, SmolLM2). Required when the workflow is part of branch protection.
- GPU E2E Tests -- full end-to-end pipeline tests. Informational -- failures produce a warning but don't block merge.

Manual dispatch includes a `suite` dropdown: `all`, `smoke`, or `e2e`. Manual runs create workflow runs for the selected branch, but they do not post a PR status check. When PR GPU testing is re-enabled, copy-pr-bot can push the current HEAD to `pull-request/<number>`, fire `gpu-tests.yml`, and post the `GPU CI Status` check result back to the PR.

To trigger from the CLI instead (no PR status check):

```bash
gh workflow run gpu-tests.yml --ref <your-branch> -f suite=all
gh workflow run gpu-tests.yml --ref <your-branch> -f suite=smoke
gh workflow run gpu-tests.yml --ref <your-branch> -f suite=e2e
```

### Test Requirements

Before submitting a PR:

- All existing tests pass (`mise run test`)
- New features include tests
- Bug fixes include regression tests

## Code Style

For detailed style guidelines covering Python, markdown, Dockerfiles, shell scripts, testing, and docstrings, see [STYLE_GUIDE.md](STYLE_GUIDE.md).

### Python Version Compatibility

Although the default development/runtime interpreter is Python 3.13, source code must remain Python 3.11 syntax-compatible until the NMP platform moves its base Python version to 3.12. Do not use Python 3.12-only syntax such as PEP 695 `type` statements or bracketed generic class/function parameters in shared package code yet.

### Formatting, Linting, and Type Checking

Use mise tasks instead of running `ruff` or `ty` directly. The tasks use pinned tool versions from `.mise.[toml|lock]` (installed via `make setup`) and check all tracked files.

```bash
mise run format   # auto-fix: dprint TOML + ruff format/import sorting + copyright headers
mise run check    # read-only local quality checks (format + lint + typecheck + copyright)
mise run test     # unit tests
# or just
mise run format && mise run check && mise run test
```

We use `dprint` for TOML, `ruff` for Python formatting and linting, and `ty` for type checking, wrapped with settings for consistency.

CI calls the same tools through atomic read-only mise tasks. Declarative tasks live in `.mise/tasks/*.toml`; bash-heavy tasks are executable file tasks under `.mise/tasks/`. Shared shell helpers live in `.mise/tasks/_lib.sh`, which is sourced by file tasks but is not executable and does not appear in `mise tasks`. `mise run check` replicates format-check + typecheck locally; `mise run validate` runs the broader pre-PR graph (`check`, `lock-check`, and `test:ci`). Pre-commit hooks (`pre-commit install`) provide faster feedback by checking only staged files, but are not a substitute for the mise tasks.

Useful task graph commands:

```bash
mise tasks                # public tasks
mise tasks --hidden       # helper and legacy alias tasks
mise tasks deps check     # inspect the quality-check graph
mise tasks deps validate  # inspect the pre-PR validation graph
```

You can also run tools directly on specific files:

```bash
bash tools/codestyle/format.sh --check src/nemo_safe_synthesizer/cli/run.py
bash tools/codestyle/ruff_check.sh src/nemo_safe_synthesizer/cli/run.py
```

All source files (`.py`, `.sh`, `.yaml`, `.yml`, `.md`) require SPDX copyright headers. `mise run format` adds them automatically; exclusions are listed in `.copyrightignore`.

All mise tasks check the entire project. Pre-commit scopes checks to staged files.

| Check | CI task | `mise run format` / `mise run check` | Pre-commit |
|---|---|---|---|
| dprint TOML format | `mise run format-check` | `format`: auto-fix; `check`: read-only | not run |
| ruff format + lint | `mise run format-check` | `format`: auto-fix; `check`: read-only | staged files (auto-fix) |
| ty typecheck | `mise run typecheck` | read-only | all files |
| copyright headers | `mise run format-check` | `format`: auto-fix; `check`: read-only | staged files (auto-fix) |
| generated CUDA metadata and uv lock drift | `mise run lock-check` | not checked | on `pyproject.toml` or `cuda_deps.toml` changes |
| DCO signoff | branch protection | not checked | commit-msg hook |

## Documentation

This project uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) for its documentation site, hosted at <https://nvidia-nemo.github.io/Safe-Synthesizer/>.

### Local Preview

Documentation dependencies are included in the `dev` bootstrap profile. If you already ran `mise run bootstrap-nss dev` (or `cpu`/`cuda`), you're set. Otherwise install them directly:

```bash
uv sync --group docs
```

Start a local server with live reload:

```bash
mise run docs:serve
# Browse to http://127.0.0.1:8000
```

In Cursor or VS Code Remote, the port is auto-forwarded. Check the Ports
panel (`Ctrl+Shift+P` > "Ports: Focus on Ports View") -- port 8000 will
appear with a local address you can open in the Simple Browser or your
system browser.

Build the static site (output in `site/`):

```bash
mise run docs:build
```

### Directory Layout

All documentation lives under `docs/`. The structure follows the [Diataxis](https://diataxis.fr/) framework:

| Directory | Content type | Examples |
| --- | --- | --- |
| `getting-started/` | Tutorials | Installation, quick start |
| `user-guide/` | How-tos & reference | CLI, configuration, SDK |
| `architecture/` | Explanations | Design decisions |
| `reference/` | API reference | Auto-generated (see below) |
| `dev-notes/` | Dev notes | Release notes, design posts |

### Adding or Editing a Page

1. Create or edit the `.md` file under the appropriate `docs/` subdirectory.
2. Add the page to the `nav:` section of `mkdocs.yml` so it appears in the sidebar.
3. Run `mise run docs:serve` and verify the page renders correctly.

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
| `.agents/skills/*/SKILL.md` | All agents (via skills index in `AGENTS.md`) | Domain-specific knowledge (testing, sync, typing, etc.) |
| `.cursor/skills/` | Cursor only | Symlinks to `.agents/skills/` for Cursor discoverability |

Conventions defined in `AGENTS.md` (code style, markdown style, testing, etc.) apply universally. Durable module-level guidance belongs in Python docstrings and source comments so it appears in the generated API reference; test-suite guidance belongs in `tests/TESTING.md`. Tool-specific config (`.cursor/rules/`, `CLAUDE.md`) reinforces those conventions for its respective tool.

Before contributing, run `mise run format` and `mise run check`. See `AGENTS.md` for full conventions.

## Releasing

Pushing a `v*` tag starts two workflows. [`release.yml`](.github/workflows/release.yml)
publishes the wheel to Test PyPI and PyPI, creates a GitHub release, and
publishes versioned documentation for final releases, including post-releases.
[`container-build.yml`](.github/workflows/container-build.yml) publishes the
CUDA image to GitHub Container Registry (GHCR).

Release versions follow [PEP 440](https://peps.python.org/pep-0440/) with major,
minor, and patch release numbers. This project uses stable releases, release
candidates, and post-releases. Prerelease versions append `rcN` without a dash;
post-releases append `.postN`. The GitHub tag always starts with a `v` prefix.

Examples:

| GitHub Tag    | PyPI Version | Valid                                           |
|:------------- |:------------ |:----------------------------------------------- |
| `v1.0.0`      | `1.0.0`      | ✅                                              |
| `v2.1.3`      | `2.1.3`      | ✅                                              |
| `v0.0.5rc0`   | `0.0.5rc0`   | ✅                                              |
| `v0.1.2rc5`   | `0.1.2rc5`   | ✅                                              |
| `v0.1.6.post1` | `0.1.6.post1` | ✅                                              |
| `1.0.0`       |              | ❌ No `v` prefix                                |
| `release-1.0` |              | ❌ Wrong format                                 |
| `v0.0.7-rc4`  |              | ❌ Dash before rc suffix                        |
| `v0.1.6-post1` |               | ❌ Dash before post-release suffix              |
| `v0.1.3a1`    |              | ❌ Alpha prereleases are not used; use rcN only |

### Release Preparation Helper

Run `mise run release:prepare -- [OPTIONS]` to compute a release tag before
creating it. The helper reads local Git tags and resolves the requested target
commit, but it does not create, delete, or push tags.

- `--bump major`, `--bump minor`, and `--bump patch` propose the corresponding
  next version's initial `rc0` tag. The default is `patch`.
- `--bump post` proposes the next `.postN` tag for the latest stable version.
- `--ref REF` selects the commit to tag and defaults to `HEAD`.
- `--json` emits the computed release plan as machine-readable JSON.

Fetch the tags you intend the helper to consider before running it. It rejects
malformed release tags, post-release tags without their stable base tag, and an
initial `rc0` proposal when an RC already exists for that version.

### Release Checklist

#### Before Publishing

- Fetch `origin/main` and tags, choose the exact release commit, and confirm its
  normal CI and manually dispatched GPU Tests run passed.
- Choose the unused tag or tags required for the release type, then record the
  exact `origin/main` SHA.
- Decide whether GHCR visibility or a separate nSpect or Pulse scan blocks the
  release. Those scans are not part of the GitHub release workflows.

After fetching tags, preview the next release's initial `rc0` version and
resolve its target commit without creating or pushing a tag:

```bash
mise run release:prepare -- --ref origin/main
```

#### Tag Trigger

Create the candidate tag at the recorded SHA and push it. This automatically
starts the package and container workflows described above.

```bash
RC_TAG=v0.1.0rc0  # Replace with the intended candidate tag.
RELEASE_SHA="$(git rev-parse 'origin/main^{commit}')"
git tag "${RC_TAG}" "${RELEASE_SHA}"
git push origin "refs/tags/${RC_TAG}"
```

Candidate tags also move the mutable GHCR `cu129` and `latest-cu129` aliases.
During candidate validation, identify the image by its immutable
`sha-<short-sha>-cu129` tag and do not treat those aliases as stable.

Never move a published tag. If code changes, create and validate the next
`rcN`.

#### Verify and Promote

- Verify the candidate tag still resolves to the recorded SHA and both
  workflows succeeded for that tag.
- Confirm the candidate exists on Test PyPI, production PyPI, and GitHub
  Releases, then pull its immutable GHCR SHA tag with the intended visibility.
- Install the production-PyPI wheel outside the repository with uv project
  configuration disabled. Check the dependency set, import, and CLI.

Use the same auxiliary indexes documented in the installation guide for the
clean CUDA install. Set `NSS_VERSION` to the candidate version:

```bash
SMOKE_DIR=/tmp/nss-release-smoke
SMOKE_VENV="${SMOKE_DIR}/.venv"
NSS_VERSION="${RC_TAG#v}"
mkdir -p "${SMOKE_DIR}"
cd "${SMOKE_DIR}"
uv --no-config venv --clear --python 3.13 "${SMOKE_VENV}"
uv --no-config pip install \
  --python "${SMOKE_VENV}/bin/python" \
  --default-index https://pypi.org/simple \
  --index https://flashinfer.ai/whl/cu129 \
  --index https://flashinfer.ai/whl/ \
  --index https://download.pytorch.org/whl/cu129 \
  --index https://wheels.vllm.ai/0.26.0/cu129 \
  --index-strategy unsafe-best-match \
  "nemo-safe-synthesizer[cu129,engine]==${NSS_VERSION}"
uv --no-config pip check --python "${SMOKE_VENV}/bin/python"
"${SMOKE_VENV}/bin/python" -c 'import nemo_safe_synthesizer'
"${SMOKE_VENV}/bin/safe-synthesizer" --help
```

Promote only after every candidate check passes. The stable tag must point to
the same tested SHA, not a later `main` commit.

```bash
STABLE_TAG="${RC_TAG%%rc*}"
git tag "${STABLE_TAG}" "${RELEASE_SHA}"
git push origin "refs/tags/${STABLE_TAG}"
```

#### Post-release

Use a post-release for a packaging or release correction that does not warrant
a new regular patch version. A post-release is final, so it does not use the
release-candidate promotion sequence. Preview the next post-release tag and
resolve its target commit without creating or pushing a tag:

```bash
mise run release:prepare -- --bump post --ref origin/main
```

After reviewing the output, create the proposed tag at the resolved commit and
push it. For example:

```bash
POST_TAG=v0.1.6.post1
RELEASE_SHA="$(git rev-parse 'origin/main^{commit}')"
git tag "${POST_TAG}" "${RELEASE_SHA}"
git push origin "refs/tags/${POST_TAG}"
```

The container workflow publishes `X.Y.Z.postN-cu129`, the immutable
`sha-<short-sha>-cu129` tag, and the mutable `cu129` and `latest-cu129` aliases.
PEP 440 post-releases do not move the shortened `X.Y-cu129` tag. Validate the
post-release wheel and immutable container tag before announcing the release.

#### After Publishing a Final Release

After publishing:

- Verify both tag-triggered workflows passed at the tested SHA.
- Confirm Test PyPI and production PyPI contain the published version.
- Confirm the GitHub release is not marked as a prerelease.
- Confirm versioned documentation is available at
  `https://nvidia-nemo.github.io/Safe-Synthesizer/<version>/`.
- Confirm GHCR exposes the expected tags with the intended visibility. Regular
  stable releases publish `X.Y.Z-cu129` and `X.Y-cu129`; post-releases publish
  `X.Y.Z.postN-cu129` without moving `X.Y-cu129`.
- Coordinate a NeMo Platform package or container pin, documentation update,
  and downstream release when Platform should consume the new version. This is
  not currently automated by the Safe Synthesizer release workflow.
- Announce the release only after artifacts and versioned documentation pass
  verification.

## NMP Integration

NeMo Safe Synthesizer is developed as a standalone package and published to PyPI and optionally published to the internal NVIDIA Artifactory. The NeMo Platform (NMP) consumes it as an external dependency.

### Publishing to Artifactory

The `publish:internal` mise task builds a wheel and uploads it to NVIDIA Artifactory:

```bash
mise run publish:internal
```

This requires `TWINE_REPOSITORY_URL`, `TWINE_USERNAME`, and `TWINE_PASSWORD`
environment variables. This is a manual action; the tag-triggered release
workflow does not publish to internal Artifactory.

### Local Development with NMP

The NMP service (`services/safe-synthesizer/pyproject.toml` in the NVIDIA internal `nmp` repo) pulls `nemo-safe-synthesizer` from the `nv-shared-pypi-local` Artifactory index. It's used with a wrapper package called `safe-synthesizer-sdk`.

When iterating on NSS changes that need to be tested in the NMP service, use the Makefile targets in the NMP repo's `services/safe-synthesizer/` directory:

```bash
# In the NMP repo, from services/safe-synthesizer/
make use-nss-local          # Build local wheel and patch pyproject.toml
make use-nss-artifactory    # Revert to Artifactory (always do this before committing)
```

See the NMP service README (`services/safe-synthesizer/README.md`) in NMP for details.

Run `mise tasks` to see all available mise tasks. The Makefile only bootstraps mise and prints deprecation messages for old `make <task>` commands; use `mise run <task>` for project tasks.

---

Thank you for contributing to NeMo Safe Synthesizer!

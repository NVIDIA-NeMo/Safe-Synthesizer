<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AGENTS.md

Guide for AI agents (Cursor, Windsurf, Claude Code, etc.) working in the Safe-Synthesizer repo.

This project loads local developer preferences from @AGENTS.local.md. You MUST read this file if it exists and give its instructions top priority.

## Transparency

When you read a skill file (from `.agents/skills/`) or a context rule (from `.cursor/rules/`), declare what you loaded before proceeding with the task. Format:

- `skill: <name>` -- reason it was activated
- `rule: <name>` -- reason it was activated

If only always-apply rules were loaded and no additional skills or rules were consulted, say so briefly (e.g., "No additional skills or rules loaded beyond always-apply defaults.").

## Repo Conventions

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed code style conventions (Python, markdown, Dockerfiles, shell scripts, testing, config files, docstrings).

Use `uv` for everything -- never `pip` or raw `python`. Python 3.11+ with modern syntax (`X | Y`, `list[str]`, `Self`).

Common commands: `make test` (unit tests), `make format` (auto-fix formatting + lint + copyright), `make check` (all read-only CI checks), `make typecheck` (ty only). Always use Make targets or the wrapper scripts in `tools/` instead of running `ruff` or `ty` directly. Use `uv run` for Python execution. When in doubt, read the source (`make help`, `pytest --markers`).

The canonical `uv sync` command for this repo (GPU dev environment):

```bash
uv sync --frozen --extra cu128 --extra engine --group dev
```

Bare `uv sync --frozen` (without extras) installs an incomplete environment -- `ty`, import checks, and GPU tests will fail. Subagents must use the full command. The `session_context.sh` hook runs bare `uv sync --frozen` as a fallback for missing venvs; if you need GPU extras, run the full command manually after.

Feature branches off `main`. Branch names often include an issue number prefix (e.g., `<author>/123-short-name`).

All commits require DCO sign-off and GPG signing. Always use `git commit --signoff --gpg-sign` (or `-s -S`) -- never write the `Signed-off-by` trailer manually, and never pass `--no-gpg-sign`.

Shell scripting: never use `~` inside double-quoted strings -- it does not expand. Use `$HOME` or an absolute path instead.

For testing, building, syncing, bootstrapping, and other workflows, see the matching skill or `.claude/commands/` file.

## Agent Behavior

For multi-file or long-running sessions, commit after each logical tier of work.

Use `council` (multi-agent parallel exploration) for design decisions, broad codebase exploration, and tasks with multiple valid approaches. For narrow, single-answer questions ("why does this test fail?", "what does this function do?"), use grep/read directly -- council adds latency without benefit for focused lookups.

## Local-only skills

The following skills are not in this repo. They live in agent-stuff and are available when agent-stuff has been bootstrapped locally:

- `skill-audit` -- audit agent transcripts and update skills
- `council` -- coordinate multi-agent parallel codebase exploration
- `deslop` -- remove AI-generated code slop
- `diagnose-deps` -- diagnose transitive dependency changes via lockfile diff
- `verify-repo-statements` -- verify claims about the repo against source
- `beautiful-prose` -- hard-edged writing style contract for clean English prose
- `bulk-edit` -- inline scripts for repetitive mechanical edits across many files

The following skills exist in both this repo and agent-stuff. The repo versions add Safe-Synthesizer-specific content on top of the base:

- `git-worktrees` -- base skill plus Cursor worktree automation, DCO/GPG signing, and venv strategy

## Hooks and worktree setup

Hook scripts live in `.cursor/hooks/` and are loaded by both Cursor (`.cursor/hooks.json`) and Claude Code (`.claude/settings.json`).

| Script | Event | Purpose |
|--------|-------|---------|
| `session_context.sh` | `sessionStart` | Reports venv state; runs `uv sync --frozen` if `.venv` absent |
| `enforce-signoff.sh` | `beforeShellExecution` / `PreToolUse(Bash)` | Blocks `git commit` without `--signoff` / `-s`; blocks missing `--gpg-sign` / `-S` |
| `audit.sh` | most events | Appends timestamped JSON to `~/.cursor/audit.log` |

Cursor parallel-agent worktrees are configured via `.cursor/worktrees.json`, which runs `.cursor/setup-worktree.sh` at worktree creation. The setup script runs `uv sync --frozen` and copies `.local.envrc` from the main worktree if present.

For manual worktrees (agent-created via `git worktree add`), run `uv sync --frozen` after creation. See the `git-worktrees` skill for the full workflow.

## Fast-model subagent suitability

Prefer a fast model (`model: fast`) when delegating focused, narrow tasks to a subagent:

- Verification -- run tests, spot-check implementation (handled by `.cursor/agents/verifier.md`)
- Read-only CI checks -- `make format-check`, `make lint`, `make typecheck`; report pass/fail
- Narrow exploration -- single module or 1-2 files
- Dependency diff -- run `tools/diff-lockfile.py` and summarize output

Use the default model for synthesis, broad exploration, and multi-step reasoning.

## Module Map

Source code lives in `src/nemo_safe_synthesizer/`:

| Path | Purpose |
|------|---------|
| `cli/` | Click CLI, main entry point |
| `config/` | Pydantic parameter models, SafeSynthesizerParameters |
| `configurator/` | Pydantic-to-Click mapping, Parameter types, validators |
| `data_processing/` | Holdout, actions, assembler, records |
| `evaluation/` | Evaluator, components (privacy, MI, AIA, PII replay), reports |
| `generation/` | GeneratorBackend, VllmBackend, regex manager, batch gen |
| `holdout/` | Train/test splitting |
| `llm/` | Model loading, metadata, memory management |
| `pii_replacer/` | NER-based PII detection and replacement |
| `privacy/` | DP transformers (Opacus integration) |
| `sdk/` | SafeSynthesizer builder, library_builder |
| `training/` | TrainingBackend, HuggingFace, Unsloth backends |
| `artifacts/` | Data quality checks, field analysis, metadata |
| `observability.py` | CategoryLogger, TracedContext, structured logging |
| `errors.py` | Custom error hierarchy -- see `diagnose-failures` skill |
| `defaults.py` | Default settings, constants (`DEFAULT_ARTIFACTS_PATH`, `PSEUDO_GROUP_COLUMN`) |
| `package_info.py` | Package version (uv-dynamic-versioning) |
| `results.py` | Result compilation (`make_nss_results`, `make_nss_summary`) |
| `utils.py` | Schema prompt creation, pattern matching helpers |

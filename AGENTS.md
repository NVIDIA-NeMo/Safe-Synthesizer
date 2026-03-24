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

## Skills

Repo-specific skills live in `.agents/skills/`. General-purpose skills (ast-nav, deslop, bulk-edit, etc.) are installed globally via `~/.cursor/skills/` and do not need repo copies.

| Skill | Purpose |
|-------|---------|
| `configurator` | Pydantic-to-Click parameter mapping, `NSSBaseModel` vs `BaseSettings`, validators |
| `diagnose-failures` | Triage test, CI, runtime, GPU, import, and type errors using the error hierarchy |
| `git-worktrees` | NSS-specific worktree overrides: venv setup, DCO/GPG signing, Cursor automation |
| `github-cli` | `gh` CLI for PRs, issues, CI workflows, code review, releases |
| `python-observability` | `CategoryLogger`, `@traced` decorators, log categories and env vars |
| `sync-agent-config` | What to update when Makefile targets, modules, markers, or skills change |
| `usage` | CLI commands, SDK builder pattern, config precedence, output layout |
| `uv-build` | `uv` package management, extras, PyTorch indexes, hatch build, versioning |

Several source modules also have their own `AGENTS.md` with internal patterns and gotchas: `cli/`, `sdk/`, `llm/`, `generation/`, `training/`, and `tests/`. When working in a subtree, check for a local `AGENTS.md` before diving in.

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

Testing gotchas: `asyncio_mode = auto` in `pytest.ini` -- async tests work without `@pytest.mark.asyncio`. The `unit_test` marker is deprecated; use `unit`. `datasets==4.3.0` is hard-pinned in `pyproject.toml` due to an unsloth incompatibility -- do not unpin it.

For testing, building, syncing, bootstrapping, and other workflows, see the matching skill or `.claude/commands/` file.

## Agent Behavior

For multi-file or long-running sessions, commit after each logical tier of work.

Use `council` (multi-agent parallel exploration) for design decisions, broad codebase exploration, and tasks with multiple valid approaches. For narrow, single-answer questions ("why does this test fail?", "what does this function do?"), use grep/read directly -- council adds latency without benefit for focused lookups.

## Hooks and worktree setup

Hook scripts live in `.cursor/hooks/` and are loaded by both Cursor (`.cursor/hooks.json`) and Claude Code (`.claude/settings.json`).

| Script | Event | Purpose |
|--------|-------|---------|
| `session_context.sh` | `sessionStart` | Reports venv state; runs `uv sync --frozen` if `.venv` absent |
| `enforce-signoff.sh` | `beforeShellExecution` / `PreToolUse(Bash)` | Blocks `git commit` without `--signoff` / `-s`; blocks missing `--gpg-sign` / `-S` |

Cursor parallel-agent worktrees are configured via `.cursor/worktrees.json`, which runs `.cursor/setup-worktree.sh` at worktree creation. The setup script runs `uv sync --frozen` and copies `.local.envrc` from the main worktree if present.

For manual worktrees (agent-created via `git worktree add`), run the full sync command after creation (not bare `uv sync --frozen` -- that omits extras). See the `git-worktrees` skill for the full workflow.

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
| `errors.py` | Error hierarchy: `SafeSynthesizerError` → `UserError` (`DataError`/`ParameterError` are also `ValueError`; `GenerationError` is also `RuntimeError`) and `InternalError` (also `RuntimeError`). See `diagnose-failures` skill |
| `defaults.py` | Default settings, constants (`DEFAULT_ARTIFACTS_PATH`, `PSEUDO_GROUP_COLUMN`) |
| `package_info.py` | Package version (uv-dynamic-versioning) |
| `results.py` | Result compilation (`make_nss_results`, `make_nss_summary`) |
| `utils.py` | Schema prompt creation, pattern matching helpers |

For component-level architecture diagrams and data flow, see [design.md](design.md).

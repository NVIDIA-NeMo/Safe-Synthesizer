<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AGENTS.md

Guide for AI agents (Cursor, Windsurf, Claude Code, etc.) working in the Safe-Synthesizer repo.

This project loads local developer preferences from @AGENTS.local.md. You MUST read this file if it exists and give its instructions top priority.

## Transparency

When you read a skill file (from `.agent/skills/`) or a context rule (from `.cursor/rules/`), declare what you loaded before proceeding with the task. Format:

- `skill: <name>` -- reason it was activated
- `rule: <name>` -- reason it was activated

If only always-apply rules were loaded and no additional skills or rules were consulted, say so briefly (e.g., "No additional skills or rules loaded beyond always-apply defaults.").

## Repo Conventions

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed code style conventions (Python, markdown, Dockerfiles, shell scripts, testing, config files, docstrings).

Use `uv` for everything -- never `pip` or raw `python`. Python 3.11+ with modern syntax (`X | Y`, `list[str]`, `Self`).

Common commands: `make test` (unit tests), `make format` (auto-fix formatting + lint + copyright), `make check` (all read-only CI checks), `make typecheck` (ty only). Always use Make targets or the wrapper scripts in `tools/` instead of running `ruff` or `ty` directly. Use `uv run` for Python execution. When in doubt, read the source (`make help`, `pytest --markers`).

Feature branches off `main`. Branch names often include an issue number prefix (e.g., `<author>/123-short-name`).

All commits require DCO sign-off. Always use `git commit --signoff` (or `-s`) -- never write the `Signed-off-by` trailer manually.

For testing, building, syncing, bootstrapping, and other workflows, see the matching skill or `.claude/commands/` file.

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

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AGENTS.md

Guide for AI agents (Cursor, Windsurf, Claude Code, etc.) working in the Safe-Synthesizer repo.

This project loads local developer preferences from @AGENTS.local.md. You MUST read this file if it exists and give its instructions top priority.

## Skills

Repo-specific skills live in `.agents/skills/`; see `.agents/README.md` for the catalog. Read a skill when the task matches its scope instead of copying workflow details into this file.

Durable implementation guidance belongs with the code it describes: function and class docstrings for public contracts and source comments for local invariants. Test-suite guidance belongs in `tests/TESTING.md`.

## Repo Conventions

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed code style conventions (Python, markdown, Dockerfiles, shell scripts, testing, config files, docstrings).

Use `uv` for everything -- never `pip` or raw `python`. Python 3.11–3.13 with modern syntax (`X | Y`, `list[str]`, `Self`). Python 3.14+ is not supported.

Common commands: `make test` (unit tests), `make format` (auto-fix formatting + lint + copyright), `make check` (all read-only CI checks), `make typecheck` (ty only). Always use Make targets or the wrapper scripts in `tools/` instead of running `ruff` or `ty` directly. Use `uv run` for Python execution. When in doubt, read the source (`make help`, `pytest --markers`).

The canonical `uv sync` command for a full GPU/dev environment is:

```bash
uv sync --frozen --extra cu128 --extra engine --group dev
```

Bare `uv sync --frozen` (without extras) installs an incomplete environment -- `ty`, import checks, and GPU tests will fail.

Feature branches off `main`. Branch names often include an issue number prefix (e.g., `<author>/123-short-name`).

Do not commit unless the user asks for a commit or PR work. When committing, all commits require DCO sign-off and GPG signing. Always use `git commit --signoff --gpg-sign` (or `-s -S`) -- never write the `Signed-off-by` trailer manually, and never pass `--no-gpg-sign`.

Shell scripting: never use `~` inside double-quoted strings -- it does not expand. Use `$HOME` or an absolute path instead.

Testing gotchas: `asyncio_mode = auto` in `pytest.ini` -- async tests work without `@pytest.mark.asyncio`. The `unit_test` marker is deprecated; use `unit`.

For testing, building, syncing, bootstrapping, worktrees, GitHub, and other recurring workflows, see the matching skill in `.agents/skills/`.

## Module Map

Source code lives in `src/nemo_safe_synthesizer/`:

| Path | Purpose |
| ---- | ------- |
| `cli/` | Click CLI, main entry point |
| `config/` | Pydantic parameter models, SafeSynthesizerParameters |
| `configurator/` | Pydantic-to-Click mapping, Parameter types, validators |
| `data_processing/` | Holdout, actions, assembler, records, shared token budget (`budget.py`), shared column validators (`validation.py`) |
| `evaluation/` | Evaluator, components (privacy, MI, AIA, PII replay), reports |
| `generation/` | GeneratorBackend, VllmBackend, regex manager, batch gen |
| `holdout/` | Train/test splitting |
| `llm/` | Model loading, metadata, memory management |
| `pii_replacer/` | NER-based PII detection and replacement |
| `privacy/` | DP transformers (Opacus integration) |
| `sdk/` | SafeSynthesizer builder, library_builder |
| `training/` | TrainingBackend, HuggingFace backend, timeseries_preprocessing (`timeseries_preprocessing.py`) |
| `artifacts/` | Data quality checks, field analysis, metadata |
| `observability.py` | CategoryLogger, TracedContext, structured logging |
| `errors.py` | Error hierarchy: `SafeSynthesizerError` → `UserError` (`DataError`/`ParameterError` are also `ValueError`; `GenerationError` is also `RuntimeError`) and `InternalError` (also `RuntimeError`). See the `safe-synthesizer` skill for user-facing runtime failure triage |
| `defaults.py` | Default settings, constants (`DEFAULT_ARTIFACTS_PATH`, `PSEUDO_GROUP_COLUMN`) |
| `package_info.py` | Package version (uv-dynamic-versioning) |
| `preflight/` | Pre-flight validation (runs against the training split produced by `Holdout`, not the full input). Package layout: `types` (dataclasses), `base` (`PreflightCheck` ABC hierarchy — `ConfigCheck`/`DataFrameCheck`/`MetadataCheck`/`AdvisoryCheck`), `registry` (`get_registry() -> PreflightRegistry`, plugin registration), `orchestrator` (`run_preflight`, `_run_registry` with dependency gating), `checks/` (15 granular core checks grouped by stage: `environment.py` for CONFIG, `dataframe.py` for DATAFRAME, `metadata.py` for METADATA, `advisory.py` for ADVISORY, plus `_helpers.py` shared helpers and public `preflight.helpers` for plugin authors). Rendering-free by design. |
| `tooling/` | Internal rendering layer. Hosts `render_preflight_report` (Rich today; agentic/plain/JSON modes planned via `RenderMode`), `PreflightRenderContext`. Intended to absorb the evaluation report renderer and alternative output modes over time. |
| `results.py` | Result compilation (`make_nss_results`, `make_nss_summary`) |
| `utils.py` | Schema prompt creation, pattern matching helpers |

For component-level architecture diagrams and data flow, see [design.md](design.md).

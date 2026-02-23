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

## Available Skills

| Skill | Location | Purpose |
|-------|----------|---------|
| `github-cli` | `.agent/skills/github-cli/` | `gh` CLI for PRs, issues, CI, releases (+ references/workflows) |
| `sync-with-nmp` | `.agent/skills/sync-with-nmp/` | Bidirectional sync with NMP (+ references/workflows) |
| `python-typing` | `.agent/skills/python-typing/` | Type hints, generics, protocols, ty checker |
| `python-testing-patterns` | `.agent/skills/python-testing-patterns/` | pytest patterns, fixtures, mocking, property-based testing |
| `python-observability` | `.agent/skills/python-observability/` | Structured logging, traced decorators, log categories |
| `python-stdlib-patterns` | `.agent/skills/python-stdlib-patterns/` | pathlib, dataclasses, functools, itertools, enum patterns |
| `docker` | `.agent/skills/docker/` | Dockerfiles, multi-stage uv builds, CUDA/GPU images, buildx bake, security |
| `pydantic` | `.agent/skills/pydantic/` | NSSBaseModel, BaseSettings, validators, ConfigDict, TypeAdapter conventions |
| `uv-build` | `.agent/skills/uv-build/` | uv deps, extras (cpu/cu128), index management, hatch build, versioning |
| `configurator` | `.agent/skills/configurator/` | Pydantic-to-Click CLI, Parameter types, conditional validators |
| `diagnose-deps` | `.agent/skills/diagnose-deps/` | Lockfile diff diagnosis, transitive dep changes |
| `deslop` | `.agent/skills/deslop/` | Remove AI-generated code slop, enforce repo style |
| `diagnose-failures` | `.agent/skills/diagnose-failures/` | Test/CI/runtime/GPU error triage |
| `sync-agent-config` | `.agent/skills/sync-agent-config/` | Keep agent config in sync with source-of-truth files |

All skills live in `.agent/skills/`. Each has a concise `SKILL.md` with quick references; skills with detailed content have a `references/` subdirectory.

## Repo Conventions

### Package Manager

Use `uv` (>=0.9.14). Never use `pip` directly.

```bash
# Install dependencies
uv sync --frozen --group dev

# Run a tool
uv run <tool>

# Add a dependency
uv add <package>
```

### Python

Python 3.11+ is required. Use modern syntax:

- `X | Y` union syntax, not `Optional[X]` or `Union[X, Y]`
- `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- `Self` type for fluent interfaces (3.11+)

### Markdown and Docstrings

- No decorative `**bold**` -- use headers, list markers, colons, and backticks for structure

### Bootstrap

```bash
# Step 1: Bootstrap tools (uv, pre-commit, etc.)
make bootstrap-tools

# Step 2: Bootstrap Python env (pick one profile)
make bootstrap-nss dev        # dev tools only
make bootstrap-nss cpu       # + engine + CPU PyTorch
make bootstrap-nss cu128     # + engine + CUDA PyTorch
make bootstrap-nss engine    # engine only
```

### Code Quality

```bash
# Format (ruff format + ruff check --fix + copyright headers)
make format

# Lint (ruff + ty + copyright check)
make lint

# Type check only
ty check
bash tools/lint/run-ty-check.sh          # changed files only
```

Type checking uses [ty](https://github.com/astral-sh/ty) (Astral's type checker), configured in `pyproject.toml` under `[tool.ty.src]`.

Important for agents: Always use `make format` and `make lint` instead of manually reformatting or editing code for style. Prefer `make <target>` over raw underlying commands when a target exists. Use `uv run` for Python execution, never raw `python` or `pip`. When in doubt, read the source (`make help`, `pytest --markers`).

### Testing

```bash
# Unit tests (fast, excludes slow)
make test

# All tests including slow (excludes e2e)
make test-slow

# SDK-related tests only
make test-sdk-related

# CI tests in a Linux container
make test-ci-container
```

Test runner: `uv run --frozen pytest -n auto --dist loadscope -vv`

Markers (defined in `pytest.ini`):
- `unit` -- unit tests (default, no marker needed)
- `slow` -- long-running tests
- `e2e` -- end-to-end pipeline tests (requires CUDA)
- `gpu_integration` -- GPU integration tests
- `integration` -- integration tests
- `infrastructure` -- infrastructure compatibility tests

Config: `pytest.ini` (asyncio_mode=auto, timeout=300s, parallel with -n 8, strict-markers).

### NMP Synchronization

```bash
# Sync a specific NMP MR
make synchronize-from-nmp-mr MR=<number>

# Full sync (requires NMP_REPO_PATH)
make synchronize-from-nmp     # NMP → Safe-Synthesizer
make synchronize-to-nmp       # Safe-Synthesizer → NMP
```

### Documentation

```bash
make docs-serve    # Local dev server with live reload
make docs-build    # Build static site
```

### Build

```bash
make build-wheel        # Build wheel (version from git tag)
make publish-internal   # Publish to NVIDIA Artifactory
```

### Branching

Feature branches off `main`. Branch names often include an issue number prefix (e.g., `<author>/123-short-name`).

---

## Using Safe-Synthesizer

This section is for agents helping users run the package, not develop it.

### CLI

Entry point: `safe-synthesizer` (installed via `pip install nemo-safe-synthesizer` or `uv add nemo-safe-synthesizer`).

```bash
# Full pipeline (train + generate + evaluate)
safe-synthesizer run --config config.yaml --url data.csv

# Train only
safe-synthesizer run train --config config.yaml --url data.csv

# Generate only (needs a trained adapter)
safe-synthesizer run generate --config config.yaml --url data.csv \
    --run-path /path/to/trained/run

# Generate with auto-discovery of adapter
safe-synthesizer run generate --config config.yaml --url data.csv \
    --auto-discover-adapter --artifact-path ./safe-synthesizer-artifacts

# Validate a config file
safe-synthesizer config validate --config config.yaml

# Clean artifacts
safe-synthesizer artifacts clean
```

CLI options map to config fields with `__` as the nested separator:

```bash
safe-synthesizer run --config config.yaml --url data.csv \
    --data__holdout=0.1 \
    --training__learning_rate=0.0001 \
    --enable_synthesis=true
```

### SDK (Programmatic)

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters

# Load config and run full pipeline
config = SafeSynthesizerParameters.from_yaml("config.yaml")
synthesizer = SafeSynthesizer(config).with_data_source("data.csv")
synthesizer.run()
results = synthesizer.results

# Builder overrides
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source(df)                   # DataFrame, URL, or file path
    .with_train(learning_rate=0.0001)
    .with_generate(num_records=10000)
    .with_evaluate(enabled=True)
)

# Stepwise execution
synthesizer = SafeSynthesizer(config).with_data_source(df)
synthesizer.process_data()
synthesizer.train()
synthesizer.generate()
synthesizer.evaluate()
```

### Configuration

Sources (highest precedence first): CLI overrides > dataset registry overrides > YAML config file.

Config sections: `data`, `replace_pii`, `training`, `generation`, `privacy`, `evaluation`, `time_series`, `enable_synthesis`, `enable_replace_pii`.

Environment variables:

| Variable | Purpose |
|----------|---------|
| `NSS_ARTIFACTS_PATH` | Default artifact output path |
| `NSS_LOG_FORMAT` | `json` or `plain` |
| `NSS_LOG_FILE` | Log file path |
| `NSS_DATASET_REGISTRY` | Dataset registry YAML path/URL |
| `NSS_CONFIG` | Config file path |
| `NSS_WANDB_MODE` | WandB mode (`online`, `offline`, `disabled`) |

### Output Layout

```
<artifact-path>/<config>---<dataset>/<timestamp>/
├── safe-synthesizer-config.json
├── train/
│   ├── adapter/              # Trained PEFT adapter
│   └── safe-synthesizer-config.json
├── generate/
│   ├── synthetic_data.csv
│   ├── evaluation_report.html
│   └── logs.jsonl
└── dataset/
    ├── training.csv
    ├── test.csv
    └── validation.csv
```

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
| `errors.py` | Custom error hierarchy (see below) |

## Error Hierarchy

Source: `src/nemo_safe_synthesizer/errors.py`

| Error Class | Base | Typical Cause | Agent Action |
|-------------|------|---------------|--------------|
| `DataError` | `UserError`, `ValueError` | Bad data (NaNs, unsupported types) | Check data/fixtures |
| `ParameterError` | `UserError`, `ValueError` | Invalid config/params | Check config fields |
| `GenerationError` | `UserError`, `RuntimeError` | Sampling failures | Check generation config/mocks |
| `InternalError` | `SafeSynthesizerError`, `RuntimeError` | Library bug | Report/fix |

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AGENTS.md

Guide for AI agents (Cursor, Windsurf, Claude Code, etc.) working in the Safe-Synthesizer repo.

## Available Skills

| Skill | Location | Purpose |
|-------|----------|---------|
| `github-cli` | `.cursor/skills/github-cli/` | `gh` CLI for PRs, issues, CI, releases (+ references/workflows) |
| `glab` | `.cursor/skills/glab/` | `glab` CLI for MRs, pipelines, approvals (+ references/gitlab-inspect) |
| `sync-with-nmp` | `.cursor/skills/sync-with-nmp/` | Bidirectional sync with NMP (+ references/workflows) |
| `python-typing` | `.cursor/skills/python-typing/` | Type hints, generics, protocols, ty checker |
| `python-testing-patterns` | `.cursor/skills/python-testing-patterns/` | pytest patterns, fixtures, mocking, property-based testing |
| `python-observability` | `.cursor/skills/python-observability/` | Structured logging, traced decorators, log categories |
| `python-stdlib-patterns` | `.cursor/skills/python-stdlib-patterns/` | pathlib, dataclasses, functools, itertools, enum patterns |
| `docker` | `.cursor/skills/docker/` | Dockerfiles, multi-stage uv builds, CUDA/GPU images, buildx bake, security |
| `pydantic` | `.cursor/skills/pydantic/` | NSSBaseModel, BaseSettings, validators, ConfigDict, TypeAdapter conventions |
| `uv-build` | `.cursor/skills/uv-build/` | uv deps, extras (cpu/cu128), index management, hatch build, versioning |
| `configurator` | `.cursor/skills/configurator/` | Pydantic-to-Click CLI, Parameter types, conditional validators |

All skills live in `.cursor/skills/`. Each has a concise `SKILL.md` with quick references; skills with detailed content have a `references/` subdirectory.

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

### Bootstrap

```bash
# Full dev environment (tools + Python + deps)
make bootstrap-dev-env

# Just Python deps (pick one)
make bootstrap-nss dev        # dev tools only
make bootstrap-nss cpu        # + engine + CPU PyTorch
make bootstrap-nss cu128      # + engine + CUDA PyTorch
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

**Important for agents**: Always use `make format` and `make lint` instead of manually reformatting or editing code for style. These tools handle formatting (ruff), import sorting, copyright headers, and lint fixes automatically. After making code changes, run `make format` to fix style, then `make lint` to verify. Do not manually rewrite code just to fix formatting, imports, or lint -- let the tools do it.

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

**Markers** (defined in `pytest.ini`):
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

This section is for agents helping users **run** the package, not develop it.

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

**Sources** (highest precedence first): CLI overrides > dataset registry overrides > YAML config file.

**Config sections**: `data`, `replace_pii`, `training`, `generation`, `privacy`, `evaluation`, `time_series`, `enable_synthesis`, `enable_replace_pii`.

**Environment variables**:

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

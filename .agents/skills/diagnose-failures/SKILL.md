# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
name: diagnose-failures
description: "Triage test, CI, runtime, GPU, import errors, and type errors. Triggers on: test failed, CI failed, pytest error, traceback, OOM, CUDA, import error, DataError, ParameterError, GenerationError, InternalError, ty check, ty error, typecheck, unresolved-attribute, unresolved-import, type error."
related-skills: [diagnose-deps]
license: Apache-2.0
---

# Diagnose Failures

Decision tree for triaging failures in the Safe-Synthesizer codebase.

## Error Hierarchy

Source: `src/nemo_safe_synthesizer/errors.py`

| Error Class | Base | Typical Cause | Agent Action |
|-------------|------|---------------|--------------|
| `DataError` | `UserError`, `ValueError` | Bad data (NaNs, unsupported types) | Check data/fixtures |
| `ParameterError` | `UserError`, `ValueError` | Invalid config/params | Check config fields |
| `GenerationError` | `UserError`, `RuntimeError` | Sampling failures | Check generation config/mocks |
| `InternalError` | `SafeSynthesizerError`, `RuntimeError` | Library bug | Report/fix in source |

## Test Failures

Run the failing test in isolation first:

```bash
uv run --frozen pytest path/to/test.py::test_name -vvs -n0
```

Then map the error class:
- `DataError` → check test data, fixtures, stub datasets in `tests/stub_datasets/`
- `ParameterError` → check config/params in test setup or `tests/conftest.py`
- `GenerationError` → check mock setup, generation config
- `InternalError` → likely a real bug in the source code
- `AssertionError` → check expected values, test logic

Conditional-assert pitfall: do not `assert field is not None` for fields that are legitimately `None` in some configurations (e.g. `validation_dataset` when holdout is disabled). Use a conditional guard instead.

### Pytest Markers

Markers are defined in `pytest.ini`. Run `make test` (unit), `make test-smoke`, `make test-slow` (all), or `uv run pytest -m <marker>` for specific markers.

## Type Errors (`ty`)

Always run `make typecheck` locally -- not `run-ty-check.sh` directly. The script uses `git diff --cached` (staged files only), which silently checks nothing unless files are staged. `make typecheck` runs against all source files.

```bash
make typecheck
```

Common `ty` error patterns:

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `unresolved-import` | Missing extra in venv | Run `uv sync --frozen --extra cu128 --extra engine --group dev` |
| `unresolved-attribute` | Computed property treated as config field | Check if the attribute is a `@property`, not a Pydantic field |
| `possibly-unbound` | Variable assigned only in one branch | Add an `else` branch or initialise before the conditional |
| `invalid-argument-type` | Wrong type passed to function | Check the function signature; use `cast()` only as a last resort |

Prefer fixing type errors over adding `# type: ignore`. Use `# type: ignore[<code>]` only when the error is a known `ty` false positive and fixing it would require changing correct code.

## CI Pipeline Failures

Map GitHub Actions job names to local commands:

| CI Job | Local Command |
|--------|---------------|
| `Format` | `make format` (fix) or `make format-check` (check) |
| Format (lock) | `make lock-check` |
| `Typecheck` | `make typecheck` |
| `Unit Tests` | `make test-ci` or `make test` |
| `Config-Dataset e2e` | `make test-nss-<config>-<dataset>-ci` |

If format/typecheck/unit-test were skipped: path filtering runs only when `src/` or `tests/` change. For workflow-only or docs-only PRs, run `make check` and `make test` locally or add a trivial Python change to trigger CI.

Fetch CI logs:
```bash
gh run view <run-id> --log-failed
```

## Import / Dependency Errors

- Check if the import requires an extras gate: `cpu`, `cu128`, or `engine`
- Common: `vllm`, `torch`, `unsloth` need `cpu` or `cu128` extra
- Use the `diagnose-deps` skill for lockfile diff diagnosis after `uv lock`
- Run: `uv run tools/diff-lockfile.py` to see what changed

## GPU / CUDA Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `CUDA out of memory` | Batch too large or model too big | Reduce `batch_size` or use quantization |
| `CUDA not available` | Wrong extra installed | Reinstall with `make bootstrap-nss cu128` |
| `NCCL error` | Multi-GPU issues | Use `CUDA_VISIBLE_DEVICES=0` for single-GPU |

### Running GPU / e2e Tests

- Always use `-n 0` (disables xdist forking). xdist forks crash with CUDA contexts.
- Agent sandbox blocks CUDA and network -- use `required_permissions: ["all"]`.
- Flash Attention requires head_dim >= 64. Tiny test models (e.g. GPT-2) fail with it -- pass `attn_implementation="eager"`.
- DP `max_compositions` mismatch: usually a `prv_accountant` version issue -- see `diagnose-deps` skill.

## Runtime Errors

Enable debug logging:
```bash
NSS_LOG_LEVEL=DEBUG NSS_LOG_FORMAT=plain safe-synthesizer run ...
```

Log categories:
- `user` -- user-relevant progress and results
- `runtime` -- internal operational details
- `system` -- system-level events (startup, shutdown)
- `backend` -- logs from dependencies (torch, transformers, etc.)

Artifact logs: `<workdir>/logs/` contains JSON log files for full trace.

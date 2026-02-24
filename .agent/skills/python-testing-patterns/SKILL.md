---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-testing-patterns
description: "Python testing patterns and best practices using pytest, mocking, and property-based testing. Triggers on: unit test, integration test, pytest, fixture, mock, parametrize, hypothesis, TDD, test coverage."
compatibility: "Python 3.11+, pytest with xdist and asyncio_mode=auto."
---

# Python Testing Patterns

## Running Tests

```bash
# Make targets (preferred)
make test                 # Unit tests, excluding slow
make test-slow            # All tests including slow, excluding e2e
make test-ci-container    # CI tests in a Linux container

# Direct pytest (via uv)
uv run pytest                       # Run all tests
uv run pytest -v tests/unit/        # Specific directory
uv run pytest -k "test_user"        # Match pattern
uv run pytest -m "not slow"         # Exclude slow tests
```

## pytest.ini Configuration

Key settings in `pytest.ini` at the repo root:

- `asyncio_mode = auto` -- no need for `@pytest.mark.asyncio` on every test
- `timeout = 300` -- 5-minute timeout per test
- `-n 8 --maxprocesses=8` -- parallel execution via pytest-xdist
- `--strict-markers` -- typos in marker names cause errors

## Markers

Defined in `pytest.ini`. All custom markers must be registered there.

```python
@pytest.mark.unit           # Unit tests (default, no marker needed)
@pytest.mark.slow           # Long-running tests
@pytest.mark.e2e            # End-to-end pipeline tests (requires CUDA)
@pytest.mark.gpu_integration  # GPU integration tests
@pytest.mark.integration    # Integration tests
@pytest.mark.infrastructure # Infrastructure compatibility tests
@pytest.mark.noautouse      # Skip autouse fixtures for this test
```

## Constraints

### MUST DO

- Run tests via `uv run pytest` or `make test` targets -- never bare `pytest`
- Use markers from `pytest.ini` -- register new markers before using them
- Use `conftest.py` for shared fixtures
- Use `tmp_path` fixture for file operations, never write to the repo tree
- Install test dependencies via `uv sync --group dev`
- Mark CUDA-dependent tests with `@pytest.mark.e2e` or `@pytest.mark.gpu_integration`

### MUST NOT DO

- Run `pytest` directly without `uv run`
- Add new markers without registering them in `pytest.ini`
- Use `time.sleep()` in tests -- use `pytest-timeout` or mock time
- Mock internal implementation details -- mock only external boundaries
- Write tests that depend on execution order or global state

## Repo-Specific Files

- `pytest.ini` -- test configuration (markers, timeouts, parallelism, asyncio)
- `Makefile` -- test Make targets (`make test`, `make test-slow`, etc.)
- `pyproject.toml` `[dependency-groups] test` -- test dependencies

---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-testing-patterns
description: "Python testing patterns and best practices using pytest, mocking, and property-based testing. Triggers on: unit test, integration test, pytest, fixture, mock, parametrize, hypothesis, TDD, test coverage."
compatibility: "Python 3.11+, pytest with xdist and asyncio_mode=auto."
allowed-tools: "Read Write"
depends-on: []
related-skills: [python-typing, python-observability, python-stdlib-patterns]
---

# Python Testing Patterns

Comprehensive guide to implementing robust testing strategies in Python using pytest, fixtures, mocking, parameterization, and property-based testing.

## When to Use This Skill

- Writing unit tests for Python functions and classes
- Setting up comprehensive test suites and infrastructure
- Implementing test-driven development (TDD) workflows
- Creating integration tests for APIs, databases, and services
- Mocking external dependencies and third-party services
- Testing async code and concurrent operations
- Implementing property-based testing with Hypothesis
- Setting up CI/CD test automation
- Debugging failing tests and improving test coverage

## Core Concepts

**Test Discovery**: Files matching `test_*.py` or `*_test.py`, functions starting with `test_`

**Fixtures**: Reusable test resources with setup and teardown
- Scopes: `function` (default), `class`, `module`, `session`
- Composition: Build complex fixtures from simple ones
- Share via `conftest.py` for project-wide availability

**Assertions**: Use `assert` statements, `pytest.raises()` for exceptions

**Organization**: Separate `unit/`, `integration/`, `e2e/` directories

## Quick Reference

Load detailed references for specific topics:

| Task | Reference File |
|------|----------------|
| Pytest basics, test structure, AAA pattern | `./references/pytest-fundamentals.md` |
| Fixtures, scopes, setup/teardown, conftest.py | `./references/fixtures.md` |
| Parametrization, multiple test cases | `./references/parametrized-tests.md` |
| Mocking, patching, unittest.mock, pytest-mock | `./references/mocking.md` |
| Async tests, pytest-asyncio, event loops | `./references/async-testing.md` |
| Property-based testing, Hypothesis, strategies | `./references/property-based-testing.md` |
| Monkeypatch, environment variables, attributes | `./references/monkeypatch.md` |
| Test structure, markers, conftest.py patterns | `./references/test-organization.md` |
| Coverage measurement, reports, thresholds | `./references/coverage.md` |
| Database, API, Redis, message queue testing | `./references/integration-testing.md` |
| Best practices, test quality, fixture design | `./references/best-practices.md` |

## Workflow

### 1. Basic Test Setup
```python
# test_example.py
import pytest

def test_something():
    """Descriptive test name."""
    # Arrange
    expected = 5

    # Act
    result = 2 + 3

    # Assert
    assert result == expected
```

**Run tests:**
```bash
# Repo Make targets (preferred)
make test                 # Unit tests, excluding slow
make test-slow            # All tests including slow, excluding e2e
make test-sdk-related     # Config/SDK/CLI/API tests only
make test-ci-container    # CI tests in a Linux container

# Direct pytest (via uv)
uv run pytest                       # Run all tests
uv run pytest -v tests/unit/        # Specific directory
uv run pytest -k "test_user"        # Match pattern
uv run pytest -m unit               # Run marked tests
uv run pytest -m "not slow"         # Exclude slow tests
```

### 2. Using Fixtures
```python
@pytest.fixture
def sample_data():
    """Provide test data."""
    data = {"key": "value"}
    yield data
    # Cleanup if needed

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

### 3. Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert input ** 2 == expected
```

### 4. Mocking External Dependencies
```python
from unittest.mock import patch

@patch("module.external_api_call")
def test_with_mock(mock_api):
    mock_api.return_value = {"status": "ok"}

    result = my_function()

    assert result["status"] == "ok"
    mock_api.assert_called_once()
```

### 5. Coverage Measurement
```bash
uv run pytest --cov=src --cov-report=term-missing
uv run pytest --cov=src --cov-report=html
uv run pytest --cov=src --cov-fail-under=80
```

### 6. Test Configuration

This repo's config is in `pytest.ini` at the repo root. Key settings:
- `asyncio_mode = auto` (no need for `@pytest.mark.asyncio` on every test)
- `timeout = 300` (5-minute timeout per test)
- `-n 8 --maxprocesses=8` (parallel execution via pytest-xdist)
- `--strict-markers` (typos in markers cause errors)

**Markers** (from `pytest.ini`):
```python
@pytest.mark.unit           # Unit tests (default, no marker needed)
@pytest.mark.slow           # Long-running tests
@pytest.mark.e2e            # End-to-end pipeline tests (requires CUDA)
@pytest.mark.gpu_integration  # GPU integration tests
@pytest.mark.integration    # Integration tests
@pytest.mark.infrastructure # Infrastructure compatibility tests
@pytest.mark.noautouse      # Skip autouse fixtures for this test
```

**Test dependencies** come from `uv sync --group dev` (see `[dependency-groups] test` in `pyproject.toml`).

## Constraints

### MUST DO

- Run tests via `uv run pytest` or `make test` targets -- never bare `pytest`
- Use markers from `pytest.ini` (`@pytest.mark.slow`, `@pytest.mark.e2e`, etc.)
- Use `--strict-markers` (already configured) -- typos in marker names cause errors
- Use `conftest.py` for shared fixtures, not repeated setup code
- Follow the AAA pattern (Arrange, Act, Assert) in every test
- Use descriptive test names: `test_<behavior>_<condition>_<expected>`
- Use `tmp_path` fixture for file operations, never write to the repo tree
- Install test dependencies via `uv sync --group dev`

### MUST NOT DO

- Run `pytest` directly without `uv run` -- dependencies may not resolve
- Add new markers without registering them in `pytest.ini`
- Skip `@pytest.mark.asyncio` if `asyncio_mode=auto` is set (it is in this repo -- the mark is optional)
- Use `time.sleep()` in tests -- use `pytest-timeout` or mock time
- Mock internal implementation details -- mock only external boundaries
- Write tests that depend on execution order or global state
- Commit tests that require CUDA without marking them `@pytest.mark.e2e` or `@pytest.mark.gpu_integration`

## Common Patterns

**Exception testing:**
```python
with pytest.raises(ValueError, match="error message"):
    function_that_raises()
```

**Async testing:**
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```

**Temporary files:**
```python
def test_file_operation(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    assert test_file.read_text() == "content"
```

**Markers for test selection:**
```python
@pytest.mark.slow
@pytest.mark.integration
def test_database_operation():
    pass

@pytest.mark.e2e
def test_full_pipeline():
    pass
```

## Common Mistakes

1. **Not using fixtures**: Repeating setup code across tests
   - Solution: Create fixtures in conftest.py

2. **Tests depending on order**: Global state pollution
   - Solution: Ensure test independence with proper fixtures

3. **Over-mocking**: Mocking internal implementation
   - Solution: Mock only external boundaries (APIs, databases)

4. **Missing edge cases**: Only testing happy path
   - Solution: Test boundary conditions, errors, and invalid inputs

5. **Slow tests**: Running full integration tests frequently
   - Solution: Separate unit/integration, use markers, optimize fixtures

6. **Ignoring coverage gaps**: Not measuring test coverage
   - Solution: Use pytest-cov and track metrics

7. **Poor test names**: Generic names like `test_1()`
   - Solution: Use descriptive names: `test_<behavior>_<condition>_<expected>`

8. **No cleanup**: Resources not released
   - Solution: Use fixtures with proper teardown (yield pattern)

## Resources

- **pytest**: https://docs.pytest.org/
- **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html
- **pytest-asyncio**: Testing async code (asyncio_mode=auto in this repo)
- **pytest-cov**: Coverage reporting
- **pytest-mock**: pytest wrapper for mock
- **Hypothesis**: https://hypothesis.readthedocs.io/
- **pytest-xdist**: Parallel test execution (used by default in this repo)
- **pytest-timeout**: Per-test timeouts (300s default in this repo)
- **testcontainers**: Docker containers for testing

## Repo-Specific Files

- `pytest.ini` - Test configuration (markers, timeouts, parallelism, asyncio)
- `Makefile` - Test Make targets (`make test`, `make test-slow`, etc.)
- `pyproject.toml` `[dependency-groups] test` - Test dependencies

---

## See Also

**Related Skills:**

- `python-typing` - Type hints, generics, protocols, ty checker
- `python-observability` - Structured logging, traced decorators, log categories
- `python-stdlib-patterns` - pathlib, dataclasses, functools, itertools, enum patterns

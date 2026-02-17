---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-observability
description: "Structured logging and tracing patterns for Safe Synthesizer. Triggers on: logging, logger, structlog, traced, observability, log level, log category, get_logger."
compatibility: "Python 3.11+, structlog, pydantic-settings."
allowed-tools: "Read Write"
depends-on: []
related-skills: [python-typing, python-testing-patterns, python-stdlib-patterns]
---

# Python Observability

Structured logging and tracing patterns for the Safe Synthesizer codebase. All observability is centralized in `src/nemo_safe_synthesizer/observability.py`.

## When to Use This Skill

- Adding logging to a new or existing module
- Decorating functions with trace instrumentation
- Choosing the correct log category (RUNTIME, USER, SYSTEM, BACKEND)
- Configuring log output (format, level, file, color)
- Debugging production issues via structured logs

## Core Architecture

Logging is built on **structlog** with a stdlib backend. Key components:

- `get_logger(name)` -- returns a `CategoryLogger` with category-aware sub-loggers
- `CategoryLogger` -- wraps a stdlib logger with `.runtime`, `.user`, `.system`, `.backend` adapters
- `LogCategory` -- enum: `RUNTIME`, `USER`, `SYSTEM`, `BACKEND`
- `traced` / `traced_user` / `traced_runtime` / `traced_system` / `traced_backend` -- decorator/context-manager for entry/exit logging with duration
- `initialize_observability()` -- must be called by entry points (CLI, scripts) before logging

**Important:** Logging is NOT auto-initialized on import. When used as a library, `get_logger()` returns basic stdlib loggers that integrate with the parent application's logging config.

## Pattern 1: Module-Level Logger Setup

Every module that needs logging should set up a logger at module level:

```python
from nemo_safe_synthesizer.observability import get_logger

logger = get_logger(__name__)
```

Then use it throughout the module:

```python
def process_data(records: list[dict]) -> list[dict]:
    logger.info("Processing records", extra={"count": len(records)})
    # ...
    logger.debug("Filtering complete", extra={"remaining": len(filtered)})
    return filtered
```

## Pattern 2: Log Categories

Use category-specific sub-loggers to classify log output:

```python
from nemo_safe_synthesizer.observability import get_logger

logger = get_logger(__name__)

# RUNTIME: Internal operational details (memory, timings, debug info)
logger.runtime.debug("Cache hit rate", extra={"rate": 0.95})
logger.runtime.info("Memory allocated", extra={"bytes": 1024})

# USER: User-relevant progress and results
logger.user.info("Training started", extra={"epochs": 10})
logger.user.info("Generation complete", extra={"records": 1000})

# SYSTEM: System-level events (startup, shutdown, config)
logger.system.info("Configuration loaded", extra={"config": config_name})

# BACKEND: Logs from dependency operations
logger.backend.info("Model loaded", extra={"model": model_name})

# Default (no category prefix) -- routes to RUNTIME
logger.info("Some message")
```

### Category Guidelines

| Category | Use For | Examples |
|----------|---------|----------|
| `RUNTIME` | Internal details, debug info | Cache stats, memory usage, intermediate state |
| `USER` | Progress, results the user cares about | Training epochs, generation counts, evaluation scores |
| `SYSTEM` | Lifecycle events | Startup, shutdown, config load, env setup |
| `BACKEND` | Dependency/external operations | Model loading, HuggingFace calls, vLLM operations |

## Pattern 3: Traced Decorators

Use `@traced` and its variants to log function entry/exit with duration:

```python
from nemo_safe_synthesizer.observability import traced, traced_user, traced_runtime, LogCategory

# User-visible operation (progress, results)
@traced_user("training.epoch")
def train_epoch(self, epoch_num: int):
    ...

# Internal operation (default: DEBUG level)
@traced_runtime("compute_gradients")
def _compute_gradients(self):
    ...

# With explicit category and level
@traced("data_loading", category=LogCategory.USER, level="INFO")
def load_data(path: str):
    ...

# Minimal -- uses function qualname as the operation name
@traced_runtime()
def _preprocess(self):
    ...
```

Each traced call logs:

- **Entry**: `"Entering <operation_name>"` at the specified level
- **Exit**: `"Exiting <operation_name>"` with `duration_ms` in extra
- **Error**: `"Error in <operation_name>: <message>"` at ERROR level with `error_type`

### Traced as Context Manager

For tracing a block of code rather than an entire function:

```python
from nemo_safe_synthesizer.observability import traced, LogCategory

def process_pipeline(data):
    with traced("data_loading", category=LogCategory.USER):
        loaded = load_data(data)

    with traced("validation", category=LogCategory.RUNTIME):
        validate(loaded)
```

## Pattern 4: Semantic Log Levels

| Level | Purpose | Examples |
|-------|---------|----------|
| `DEBUG` | Development diagnostics, internal state | Variable values, cache lookups, intermediate results |
| `INFO` | Normal operational events | Request lifecycle, job completion, progress updates |
| `WARNING` | Recoverable anomalies | Retry attempts, fallback used, deprecation notices |
| `ERROR` | Failures needing attention | Exceptions, missing resources, operation failures |

Never log expected behavior at ERROR. A validation failure on user input is INFO or WARNING, not ERROR.

## Pattern 5: Structured Extra Fields

Always pass context via `extra={}`, not by interpolating into the message string:

```python
# GOOD -- structured, searchable, machine-readable
logger.user.info("Training complete", extra={"epochs": 10, "loss": 0.42})

# BAD -- unstructured, hard to query
logger.user.info(f"Training complete after 10 epochs with loss 0.42")
```

Common extra fields used in this codebase:

| Field | Type | Used For |
|-------|------|----------|
| `duration_ms` | `float` | Operation timing (auto-added by `@traced`) |
| `error_type` | `str` | Exception class name (auto-added by `@traced`) |
| `count` / `records` | `int` | Number of items processed |
| `model` / `model_name` | `str` | Model identifier |
| `config` / `config_name` | `str` | Configuration identifier |
| `path` / `file` | `str` | File system path |

## Pattern 6: CLI Logging Setup

Entry points must initialize observability before logging. The CLI does this via `configure_logging_from_workdir` + `initialize_observability`:

```python
from nemo_safe_synthesizer.observability import (
    configure_logging_from_workdir,
    initialize_observability,
    get_logger,
)

# 1. Configure logging destination (sets env vars for NSS_LOG_FILE, etc.)
log_file = configure_logging_from_workdir(workdir, log_level="INFO")

# 2. Initialize (reads settings, sets up structlog + handlers)
initialize_observability()

# 3. Now logging works
logger = get_logger(__name__)
logger.user.info("CLI started")
```

## Configuration

Environment variables (read by `NSSObservabilitySettings`):

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `NSS_LOG_FORMAT` | `json`, `plain` | Auto-detect (plain if tty) | Log output format |
| `NSS_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `DEBUG_DEPENDENCIES` | `INFO` | Log verbosity |
| `NSS_LOG_FILE` | File path | None | Additional JSON log file |
| `NSS_LOG_COLOR` | `true`, `false` | Auto-detect (true if tty) | Colorized console output |
| `OTEL_SERVICE_NAME` | String | `nemo-safe-synthesizer` | OpenTelemetry service name |

## Constraints

### MUST DO

- Use `get_logger(__name__)` from `nemo_safe_synthesizer.observability` -- never `logging.getLogger()` or `structlog.get_logger()` directly
- Use `@traced` / `@traced_user` / `@traced_runtime` for public API methods and significant operations
- Pass structured data via `extra={}` dict, not string interpolation
- Choose the correct `LogCategory` for the audience (USER for progress/results, RUNTIME for internals)
- Call `initialize_observability()` in entry points before any logging

### MUST NOT DO

- Import `logging` directly for creating loggers -- always go through `get_logger()`
- Log sensitive data (credentials, tokens, PII) -- use `extra={"sensitive": True}` to mark and filter
- Use `print()` for operational output -- use the logger
- Initialize observability in library code -- only entry points (CLI, scripts) call `initialize_observability()`
- Use `logger.exception()` outside of an `except` block

## Repo-Specific Files

- `src/nemo_safe_synthesizer/observability.py` - Central observability module (all logging infra)
- `src/nemo_safe_synthesizer/cli/utils.py` - CLI logging setup helpers

---

## See Also

**Related Skills:**

- `python-typing` - Type hints, generics, protocols, ty checker
- `python-testing-patterns` - pytest patterns, fixtures, mocking, property-based testing
- `python-stdlib-patterns` - pathlib, dataclasses, functools, itertools, enum patterns

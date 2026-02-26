---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-observability
description: "Structured logging and tracing patterns for Safe Synthesizer. Triggers on: logging, logger, structlog, traced, observability, log level, log category, get_logger."
compatibility: "Python 3.11+, structlog, pydantic-settings."
---

# Python Observability

See [STYLE_GUIDE.md](../../../STYLE_GUIDE.md) for the complete logging and observability conventions.

All observability is centralized in `src/nemo_safe_synthesizer/observability.py`.

## Core API

- `get_logger(name)` -- returns a `CategoryLogger` with `.runtime`, `.user`, `.system`, `.backend` sub-loggers
- `LogCategory` -- enum: `RUNTIME`, `USER`, `SYSTEM`, `BACKEND`
- `traced` / `traced_user` / `traced_runtime` / `traced_system` / `traced_backend` -- decorator/context-manager for entry/exit logging with duration
- `initialize_observability()` -- must be called by entry points (CLI, scripts) before logging

Logging is NOT auto-initialized on import. Library code just calls `get_logger()`.

## Module-Level Logger

```python
from nemo_safe_synthesizer.observability import get_logger

logger = get_logger(__name__)

logger.info("Default routes to RUNTIME")
logger.user.info("Training started", extra={"epochs": 10})
logger.runtime.debug("Cache hit", extra={"rate": 0.95})
logger.system.info("Config loaded", extra={"config": name})
logger.backend.info("Model loaded", extra={"model": model_name})
```

## Log Categories

| Category | Use For | Examples |
|----------|---------|----------|
| `RUNTIME` | Internal details, debug info | Cache stats, memory usage, intermediate state |
| `USER` | Progress, results the user cares about | Training epochs, generation counts, scores |
| `SYSTEM` | Lifecycle events | Startup, shutdown, config load |
| `BACKEND` | Dependency/external operations | Model loading, HuggingFace calls, vLLM |

## Traced Decorators

```python
from nemo_safe_synthesizer.observability import traced, traced_user, traced_runtime, LogCategory

@traced_user("training.epoch")
def train_epoch(self, epoch_num: int): ...

@traced_runtime("compute_gradients")
def _compute_gradients(self): ...

@traced("data_loading", category=LogCategory.USER, level="INFO")
def load_data(path: str): ...
```

Also works as a context manager:

```python
with traced("validation", category=LogCategory.RUNTIME):
    validate(loaded)
```

Each traced call logs entry, exit (with `duration_ms`), and errors (with `error_type`).

## CLI Initialization

Entry points must initialize before logging:

```python
from nemo_safe_synthesizer.observability import (
    configure_logging_from_workdir,
    initialize_observability,
    get_logger,
)

log_file = configure_logging_from_workdir(workdir, log_level="INFO")
initialize_observability()
logger = get_logger(__name__)
```

## Environment Variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `NSS_LOG_FORMAT` | `json`, `plain` | Auto-detect | Log output format |
| `NSS_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | Log verbosity |
| `NSS_LOG_FILE` | File path | None | Additional JSON log file |
| `NSS_LOG_COLOR` | `true`, `false` | Auto-detect | Colorized console output |

## Constraints

### MUST DO

- Use `get_logger(__name__)` -- never `logging.getLogger()` or `structlog.get_logger()` directly
- Use `@traced` / `@traced_user` / `@traced_runtime` for significant operations
- Pass structured data via `extra={}`, not string interpolation
- Choose the correct `LogCategory` (USER for progress/results, RUNTIME for internals)
- Call `initialize_observability()` in entry points before any logging

### MUST NOT DO

- Import `logging` directly for creating loggers
- Log sensitive data (credentials, tokens, PII)
- Use `print()` for operational output
- Initialize observability in library code -- only entry points call `initialize_observability()`

## Repo Files

- `src/nemo_safe_synthesizer/observability.py` -- central observability module
- `src/nemo_safe_synthesizer/cli/utils.py` -- CLI logging setup helpers

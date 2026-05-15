<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Observability

Safe Synthesizer centralizes structured logging and tracing in
`src/nemo_safe_synthesizer/observability.py`. For normative code style, see
[STYLE_GUIDE.md](../../STYLE_GUIDE.md#logging-and-observability). This page
covers the high-level model.

## Model

Package code creates loggers with `get_logger(__name__)`:

```python
from nemo_safe_synthesizer.observability import get_logger

logger = get_logger(__name__)
```

`get_logger()` returns a `CategoryLogger` with category adapters:

| Category | Use for |
| -------- | ------- |
| `runtime` | Internal operational detail, timings, memory, debug state. |
| `user` | Progress and results that matter to the person running the pipeline. |
| `system` | Startup, shutdown, configuration, and lifecycle events. |
| `backend` | Dependency or external-library activity, such as Torch, Hugging Face, vLLM, or WandB. |

Use `extra={}` for fields that downstream tooling should query. Keep secrets,
credentials, tokens, raw PII, and full data records out of logs.

## Initialization

Logging is not initialized on import. Entry points own process-wide logging
configuration; library modules do not call `initialize_observability()`.

```python
from nemo_safe_synthesizer.observability import (
    configure_logging_from_workdir,
    initialize_observability,
)

configure_logging_from_workdir(workdir, log_level="INFO")
initialize_observability()
```

## Tracing

Tracing is optional. Use `traced_*` decorators or `traced(...)` context blocks
for entry points, phase boundaries, or operations where duration and error
metadata help diagnose runs.

```python
from nemo_safe_synthesizer.observability import LogCategory, traced, traced_user


@traced_user("training.epoch")
def train_epoch(epoch_num: int) -> None:
    ...


with traced("validation", category=LogCategory.RUNTIME):
    validate(loaded)
```

Each traced block logs entry, exit with `duration_ms`, and errors with
`error_type`.

## Environment Variables

`NSSObservabilitySettings` reads logging settings from the environment or
`.env`. User-facing details live in [Environment Variables](../user-guide/environment.md).

- `NSS_LOG_FORMAT`: `json` or `plain`.
- `NSS_LOG_LEVEL`: `INFO`, `WARNING`, `ERROR`, `CRITICAL`,
  `DEBUG_DEPENDENCIES`, or `DEBUG`.
- `NSS_LOG_FILE`: optional JSON log file path.
- `NSS_LOG_COLOR`: colorized console output.
- `OTEL_SERVICE_NAME`: OpenTelemetry service name.

`DEBUG` raises Safe Synthesizer logs to debug and leaves dependency logs at
info. `DEBUG_DEPENDENCIES` also raises backend dependency logs to debug.

## Key Files

| File | Purpose |
| ---- | ------- |
| `src/nemo_safe_synthesizer/observability.py` | Category loggers, settings, structlog configuration, tracing decorators, and heartbeat helpers. |
| `src/nemo_safe_synthesizer/cli/utils.py` | CLI logging setup helpers. |
| `docs/user-guide/environment.md` | User-facing environment variable reference. |

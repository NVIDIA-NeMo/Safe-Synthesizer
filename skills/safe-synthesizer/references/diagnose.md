<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Diagnose Safe Synthesizer

Use for install failures, runtime errors, generation failures, OOM, pre-flight validation, and environment problems.

Read first:

- `docs/user-guide/troubleshooting.md` for phase-specific runtime failures.
- `docs/user-guide/environment.md` for cache, endpoint, WandB, and offline setup.
- `docs/user-guide/evaluating-data.md` for metrics, DP errors, and quality issues.

Debug rerun:

```bash
safe-synthesizer run -v --config config.yaml --data-source data.csv
```

If dependency logs are relevant:

```bash
safe-synthesizer run -vv --config config.yaml --data-source data.csv
```

Diagnosis response shape:

- Likely phase and cause.
- One verification command.
- One next fix.
- Docs path that supports the diagnosis.

Do not propose source changes unless logs or tests indicate an implementation bug.

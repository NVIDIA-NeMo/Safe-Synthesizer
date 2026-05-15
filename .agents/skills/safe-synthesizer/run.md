<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Safe Synthesizer

Use for CLI, SDK, first run, logging, offline mode, and general pipeline execution.

Read first:

- `docs/user-guide/getting-started.md` for install, prerequisites, and first run.
- `docs/user-guide/running.md` for CLI, SDK, dataset registry, logging, offline mode, and output layout.
- `docs/user-guide/environment.md` for cache, endpoint, WandB, and offline environment variables.

Default CLI:

```bash
safe-synthesizer run --config config.yaml --data-source data.csv
```

Differential privacy example:

```bash
safe-synthesizer run --config config.yaml --data-source data.csv \
  --privacy__dp_enabled true \
  --privacy__epsilon 8.0
```

SDK example:

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

synthesizer = (
    SafeSynthesizer()
    .with_data_source("data.csv")
    .with_differential_privacy(dp_enabled=True, epsilon=8.0)
)
synthesizer.run()
```

Response shape:

- Command or SDK snippet.
- Relevant docs path.
- Precedence note when overrides are used.

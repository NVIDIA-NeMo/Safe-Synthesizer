<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configure Safe Synthesizer

Use for YAML config, CLI overrides, SDK builder overrides, config validation, and parameter precedence.

Read first:

- `docs/user-guide/configuration.md` for parameter fields and precedence.
- `docs/user-guide/running.md` for combining YAML, CLI, SDK, and dataset registry inputs.
- `docs/user-guide/environment.md` for infrastructure settings controlled by environment variables.

Precedence:

- CLI: CLI flags > dataset registry overrides > YAML config file > model defaults.
- SDK: SDK builder calls > YAML config file > model defaults.

CLI nested fields use `__`:

```bash
safe-synthesizer run --config config.yaml --data-source data.csv \
  --generation__num_records 10000
```

SDK builder overrides:

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

synthesizer = (
    SafeSynthesizer()
    .with_data_source("data.csv")
    .with_generate(num_records=10000)
)
```

For implementation internals around Pydantic models, Click option generation, or validators, read `docs/developer-guide/configuration_management.md`.

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Safe Synthesizer

Use for CLI, SDK, first run, logging, offline mode, and general pipeline execution.

Read first:

- `docs/user-guide/getting-started.md` or the
  [published getting started guide](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/getting-started/)
  for install, prerequisites, and first run.
- `docs/user-guide/running.md` or the
  [published running guide](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/running/)
  for CLI, SDK, dataset registry, logging, offline mode, and output layout.
- `docs/user-guide/environment.md` or the
  [published environment guide](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/environment/)
  for cache, endpoint, WandB, and offline environment variables.

Before the first run with PII replacement (on by default), check whether the
Nemotron-Personas dataset for `replace_pii.locale` exists under
`${HOME}/.data-designer/managed-assets/datasets/` (or `$NSS_NEMOTRON_PERSONAS_PATH/datasets/`).
If it is missing, NSS replaces names with Faker instead; tell the user and offer
the download steps in the getting started guide's "Optional Nemotron-Personas
Data" section, which need an NGC API key.

Automatic PII planning and free-text PII detection need a CUDA GPU. If PyTorch
reports none from an agent sandbox, run the command where `nvidia-smi` lists the GPU rather than
assuming the machine has none.

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

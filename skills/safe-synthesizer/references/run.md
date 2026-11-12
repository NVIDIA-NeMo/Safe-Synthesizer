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
- `docs/user-guide/customize-pii-replacement.md` or the
  [published PII customization guide](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/customize-pii-replacement/)
  for reviewing the PII replacement plan, samplers, and the planning LLM.

Before the first run with PII replacement (on by default), check whether the
Nemotron-Personas dataset for `replace_pii.locale` exists under
`${HOME}/.data-designer/managed-assets/datasets/` (or `$NSS_NEMOTRON_PERSONAS_PATH/datasets/`).
If it is missing, NSS replaces names with Faker instead; tell the user and offer
the download steps in the getting started guide's "Optional Nemotron-Personas
Data" section, which need an NGC API key.

Free-text PII detection, and automatic PII planning without
`NSS_INFERENCE_ENDPOINT`, need a CUDA GPU. If PyTorch
reports none from an agent sandbox, run the command where `nvidia-smi` lists the GPU rather than
assuming the machine has none.

When the user plans or replaces PII repeatedly, such as `run replace-pii`,
`plan_pii_replacement()`, or `replace_pii()` while refining a plan, each run
starts the planning model's vLLM server again, which takes about a minute and a
half. Suggest starting one server with `vllm serve` and reusing it, following
"Reuse one server across runs" in `docs/user-guide/customize-pii-replacement.md`.
Copy the `vllm serve` flags from the bundled profile the user plans with, under
`src/nemo_safe_synthesizer/pii_replacer/local_inference/profiles/`. With an
endpoint NSS applies none of the profile's settings, so also set
`NSS_INFERENCE_TIMEOUT` and `NSS_INFERENCE_REQUEST_OPTIONS` to the profile's
`request_timeout_seconds` and `request_options`:

```bash
export NSS_INFERENCE_ENDPOINT=http://127.0.0.1:8000/v1
export NSS_INFERENCE_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
export NSS_INFERENCE_TIMEOUT=300
export NSS_INFERENCE_REQUEST_OPTIONS='{"temperature": 1.0, "top_p": 0.95, "thinking_token_budget": 4000}'
```

The server holds most of the GPU until stopped. Tell the user to stop it and
unset `NSS_INFERENCE_ENDPOINT` before training or generating on the same GPU.

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

synthesizer = SafeSynthesizer().with_data_source("data.csv").with_differential_privacy(dp_enabled=True, epsilon=8.0)
synthesizer.run()
```

Response shape:

- Command or SDK snippet.
- Relevant docs path.
- Precedence note when overrides are used.

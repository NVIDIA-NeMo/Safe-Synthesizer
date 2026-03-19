---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: usage
description: "How to run Safe-Synthesizer: CLI commands, SDK builder pattern, configuration, environment variables, and output layout. Triggers on: safe-synthesizer run, CLI, SDK, pipeline, config, generate, train, evaluate, SafeSynthesizer, with_data_source, from_yaml."
---

# Using Safe-Synthesizer

Reference for agents helping users run the package (not develop it).

## CLI

Entry point: `safe-synthesizer` (installed via `pip install nemo-safe-synthesizer` or `uv add nemo-safe-synthesizer`).

```bash
# Full pipeline (train + generate + evaluate)
safe-synthesizer run --config config.yaml --data-source data.csv

# Train only
safe-synthesizer run train --config config.yaml --data-source data.csv

# Generate only (needs a trained adapter)
safe-synthesizer run generate --config config.yaml --data-source data.csv \
    --run-path /path/to/trained/run

# Generate with auto-discovery of adapter
safe-synthesizer run generate --config config.yaml --data-source data.csv \
    --auto-discover-adapter --artifact-path ./safe-synthesizer-artifacts

# Validate a config file
safe-synthesizer config validate --config config.yaml

# Clean artifacts
safe-synthesizer artifacts clean
```

CLI options map to config fields with `__` as the nested separator:

```bash
safe-synthesizer run --config config.yaml --data-source data.csv \
    --data__holdout=0.1 \
    --training__learning_rate=0.0001
```

## SDK (Programmatic)

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters

# Load config and run full pipeline
config = SafeSynthesizerParameters.from_yaml("config.yaml")
synthesizer = SafeSynthesizer(config).with_data_source("data.csv")
synthesizer.run()
results = synthesizer.results

# Builder overrides
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source(df)                   # DataFrame, URL, or file path
    .with_train(learning_rate=0.0001)
    .with_generate(num_records=10000)
    .with_evaluate(enabled=True)
)

# Stepwise execution
synthesizer = SafeSynthesizer(config).with_data_source(df)
synthesizer.process_data()
synthesizer.train()
synthesizer.generate()
synthesizer.evaluate()
```

## Configuration

Exactly what avenues of configuration are available, and thus how precedence is resolved, depends on how you run the pipeline. Settings are resolved in this order, from highest (first) to lowest priority (last).

- CLI: CLI flags > dataset registry overrides > YAML config file > defaults
- SDK: Python SDK builder calls > YAML config file > defaults

Config sections: `data`, `replace_pii`, `training`, `generation`, `privacy`, `evaluation`, `time_series`.

Environment variables:

| Variable | Purpose |
|----------|---------|
| `NSS_ARTIFACTS_PATH` | Default artifact output path |
| `NSS_LOG_FORMAT` | `json` or `plain` |
| `NSS_LOG_FILE` | Log file path |
| `NSS_DATASET_REGISTRY` | Dataset registry YAML path/URL |
| `NSS_CONFIG` | Config file path |
| `NSS_WANDB_MODE` | WandB mode (`online`, `offline`, `disabled`) |

## Output Layout

```
<artifact-path>/<config>---<dataset>/<timestamp>/
├── safe-synthesizer-config.json
├── train/
│   ├── adapter/              # Trained PEFT adapter
│   └── safe-synthesizer-config.json
├── generate/
│   ├── synthetic_data.csv
│   ├── evaluation_report.html
│   └── logs.jsonl
└── dataset/
    ├── training.csv
    ├── test.csv
    └── validation.csv
```

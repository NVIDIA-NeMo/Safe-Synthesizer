<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

PII replacement v3 uses a dataset-specific replacement plan. The plan names
the columns NSS should replace, the entity type in each column, optional format
patterns, and dependencies between related columns.

This branch provides the configuration and plan-resolution contract plus
plan-only CLI and SDK workflows. Replacement execution remains deferred. Until
the executor is available, set `replace_pii: null`, pass `--no-replace-pii`, or
call `.with_replace_pii(enable=False)` to run the synthesis pipeline.

## Replacement plan sources

`replace_pii.replacement_plan` accepts three forms.

The `replace_pii` configuration has an integer `schema_version`. This release
accepts version `3`; an omitted version is interpreted as version `3`. NSS
includes the version whenever it serializes the configuration.

### Automatic discovery

Use `auto_discovery` to run the heuristic plan discoverer. When `llm` is
configured, NSS passes the heuristic result to the LLM plan enhancer before
validating the final plan.

```yaml
replace_pii:
  schema_version: 3
  replacement_plan: auto_discovery
```

### Inline plan

An inline plan is written directly under `replacement_plan` in the main NSS
configuration:

```yaml
replace_pii:
  schema_version: 3
  replacement_plan:
    scope: dataframe
    columns_to_replace:
      - column_name: full_name
        entity_type: full_name
        pattern: "{First} {Last}"
      - column_name: email
        entity_type: email
        pattern: "{f}.{last}@{domain}"
        depends_on:
          - column_name: full_name
```

### Plan file

A plan file is a separately versioned YAML document containing
`schema_version` followed by the same fields as an inline plan:

```yaml
schema_version: 3
scope: dataframe
columns_to_replace:
  - column_name: email
    entity_type: email
```

Set `replacement_plan` to its path:

```yaml
replace_pii:
  schema_version: 3
  replacement_plan: ./pii_replacement_plan.yaml
```

Inline plans and plan files are authoritative: NSS validates them against the
input dataframe but does not run heuristic or LLM discovery. This bypass applies
only to plan discovery. If `llm` is configured, the replacement executor can
still use it to replace PII found inside free-text columns named by the plan.

## Plan-only workflow

Resolve and save a plan from the full input dataframe without running holdout,
model metadata, replacement, training, generation, or evaluation:

```bash
safe-synthesizer run replace-pii --plan-only \
  --config config.yaml \
  --data-source data.csv \
  --run-path ./pii-plan
```

The command writes `./pii-plan/pii_replacement_plan.yaml`. Without
`--run-path`, it writes the same filename in the standard timestamped NSS run
directory under `--artifact-path`.

The matching SDK interface returns the resolved plan and writes YAML only when
an output path is supplied:

```python
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

config = SafeSynthesizerParameters.from_yaml("config.yaml")
plan = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .plan_pii_replacement("pii_replacement_plan.yaml")
)
```

The generated standalone plan can be reviewed, edited, and reused as
`replace_pii.replacement_plan` in a later run.

## LLM-assisted planning and free-text replacement

The `llm` mapping configures the OpenAI-compatible inference service shared by
plan enhancement and free-text replacement. During automatic discovery, the LLM
enhances the heuristic plan. During execution, the same service processes
free-text columns in the resolved plan.

```yaml
replace_pii:
  schema_version: 3
  replacement_plan: auto_discovery
  llm:
    model_id: nvidia/nemotron-3-ultra-550b-a55b
    max_workers: 8
```

An empty mapping (`llm: {}`) enables the existing NSS inference defaults. Set
the OpenAI-compatible endpoint at runtime through `NSS_INFERENCE_ENDPOINT` or
the `--inference-endpoint-url` CLI option. For example, a local vLLM server may
use `NSS_INFERENCE_ENDPOINT=http://localhost:8000/v1` with its served model ID.

Supply the inference API key at runtime through `NSS_INFERENCE_KEY` or the
`--inference-api-key` CLI option. NSS does not store the key in configuration or
plan artifacts.

LLM operations can send bounded raw cell samples during plan enhancement and raw
free-text values during replacement. Do not enable them unless the endpoint is
approved to receive the input data.

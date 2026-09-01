<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

PII replacement v3 uses a dataset-specific replacement plan. The plan names
the columns NSS should replace, the entity type in each column, optional format
patterns, and dependencies between related columns.

This branch provides the configuration and plan-resolution contract. The
replacement executor and production LLM adapter will be added separately. Until
the executor is available, set `replace_pii: null`, pass `--no-replace-pii`, or
call `.with_replace_pii(enable=False)` to run the synthesis pipeline.

## Replacement plan sources

`replace_pii.replacement_plan` accepts three forms.

### Automatic discovery

Use `auto_discovery` to run the heuristic plan discoverer. When `llm` is
configured, NSS passes the heuristic result to the LLM plan enhancer before
validating the final plan.

```yaml
replace_pii:
  replacement_plan: auto_discovery
```

### Embedded plan

An embedded plan is written directly under `replacement_plan` in the main NSS
configuration:

```yaml
replace_pii:
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

A plan file is a separate YAML file containing the same mapping as an embedded
plan. Set `replacement_plan` to its path:

```yaml
replace_pii:
  replacement_plan: ./pii_replacement_plan.yaml
```

Embedded plans and plan files are authoritative: NSS validates them against the
input dataframe but does not run heuristic or LLM discovery. This bypass applies
only to plan discovery. If `llm` is configured, the replacement executor can
still use it to replace PII found inside free-text columns named by the plan.

## LLM-assisted planning and free-text replacement

The `llm` mapping configures the OpenAI-compatible inference service shared by
plan enhancement and free-text replacement. During automatic discovery, the LLM
enhances the heuristic plan. During execution, the same service processes
free-text columns in the resolved plan.

```yaml
replace_pii:
  replacement_plan: auto_discovery
  llm:
    endpoint_url: https://integrate.api.nvidia.com/v1
    model_id: nvidia/nemotron-3-ultra-550b-a55b
    max_workers: 8
```

An empty mapping (`llm: {}`) enables the existing NSS inference defaults. A
local vLLM OpenAI-compatible server can be selected with an endpoint such as
`http://localhost:8000/v1` and its served model ID.

Supply the inference API key at runtime through `NSS_INFERENCE_KEY` or the
`--inference-api-key` CLI option. NSS does not store the key in configuration or
plan artifacts.

LLM operations can send bounded raw cell samples during plan enhancement and raw
free-text values during replacement. Do not enable them unless the endpoint is
approved to receive the input data.

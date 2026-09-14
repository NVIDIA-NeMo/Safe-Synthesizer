<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

PII replacement v3 uses a dataset-specific replacement plan. The plan names
the columns NSS should replace, the entity type in each column, optional format
patterns, and dependencies between related columns.

NSS applies structured-column replacements before training. Free-text named
entity detection and span replacement are not yet part of replacement
execution.

Set `replace_pii: null`, pass `--no-replace-pii`, or call
`.with_replace_pii(enable=False)` to run the synthesis pipeline without
replacement.

## Managed person sampling

The default `managed` sampler draws names, email addresses, phone numbers, and
street-address components from the extended Nemotron Personas locale datasets.
Choose the NGC resource matching `replace_pii.replacement.locale`. Locale
resources are versioned independently, so use the version published for the
selected language and country. For example, download version `0.0.2` of the
`en_US` resource after installing and authenticating the NGC CLI:

```bash
ngc registry resource download-version \
  nvidia/nemotron-personas/nemotron-personas-dataset-en_us:0.0.2
```

Place the downloaded parquet files in the default managed-assets directory:

```bash
mkdir -p "${HOME}/.data-designer/managed-assets/datasets"
cp nemotron-personas-dataset-*/*.parquet \
  "${HOME}/.data-designer/managed-assets/datasets/"
```

The sampler loads `<managed-assets>/datasets/<locale>.parquet`. The locale in
the configuration must match the downloaded parquet filename; for the example
above, that is `en_US.parquet`:

```yaml
replace_pii:
  replacement:
    locale: en_US
  sampler:
    backend: managed
```

To store the files elsewhere, set `replace_pii.sampler.managed_assets_path` to
the directory containing `datasets/`, or set `NSS_MANAGED_ASSETS_PATH`:

```yaml
replace_pii:
  sampler:
    backend: managed
    managed_assets_path: /path/to/managed-assets
```

When an applicable locale asset or required field is unavailable, NSS warns and
uses Faker for that value. Set `backend: faker` to use Faker directly without a
managed dataset. See NVIDIA's
[person-sampling setup](https://docs.nvidia.com/nemo/datadesigner/concepts/person-sampling)
to select a locale resource and the
[`en_US` NGC resource](https://catalog.ngc.nvidia.com/orgs/nvidia/nemotron-personas/resources/nemotron-personas-dataset-en_us/-)
for the example above.

### Dependency label mappings

Dependency values are matched to sampler labels case-insensitively, so values
such as `Female` and `female` need no configuration. When the input dataset and
sampler use different label vocabularies, add sparse overrides under the
sampler:

```yaml
replace_pii:
  sampler:
    backend: managed
    dependency_value_mappings:
      gender:
        Non-binary: null
      ethnic_background:
        Asian:
          - east asian
          - south asian
          - southeast asian
        Black or African American:
          - black
```

Each nonempty list selects the union of managed-asset rows with those labels.
An explicit `null` disables that condition for the matching input value. If no
override exists, NSS uses case-insensitive identity matching and reports an
error when the managed asset has no matching candidates. Faker accepts the same
configuration and applies mappings for attributes it supports, such as gender;
unsupported attributes are ignored.

## Replacement plan sources

The `replace_pii` configuration has an integer `schema_version`. This release
accepts version `3`; an omitted version is interpreted as version `3`. NSS
includes the version whenever it serializes the configuration.

`replace_pii.replacement_plan` accepts three forms.

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
input dataframe but does not run heuristic or LLM discovery.

## Plan-only workflow

Resolve and save a plan from the full input dataframe without running holdout,
model metadata, replacement, training, generation, or evaluation:

```bash
safe-synthesizer run replace-pii --plan-only \
  --config config.yaml \
  --data-source data.csv
```

The command writes `pii_replacement_plan.yaml` in the standard timestamped NSS
run directory under `--artifact-path`.

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

## LLM-assisted planning

The `llm` mapping configures the OpenAI-compatible inference service used for
automatic plan enhancement.

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

The endpoint resolves from the explicit CLI runtime flag, then
`NSS_INFERENCE_ENDPOINT`, then the NSS default; it is never persisted in NSS
configuration. The model resolves from the explicit CLI runtime flag, then
`NSS_INFERENCE_MODEL`, `replace_pii.llm.model_id`, and finally the NSS default.
The default hosted NVIDIA endpoint requires an API key. Keyless operation is
supported for local OpenAI-compatible endpoints.

Supply the inference API key at runtime through `NSS_INFERENCE_KEY` or the
`--inference-api-key` CLI option. NSS does not store the key in configuration or
plan artifacts.

Free-text columns use GLiNER2 plus applicable deterministic built-in regex
rules:

```yaml
replace_pii:
  free_text_detection:
    model_id: fastino/gliner2.5-base-v1
    threshold: 0.3
    batch_size: 8
    chunk_length: 384
    chunk_overlap: 128
```

Automatic discovery uses two LLM passes. The first classifies every column's
semantic entity type and may propose a replacement pattern, in bounded batches
of at most 32 profiles and 48 KiB of profile evidence. Each profile contains
deterministic statistics and up to eight distinct cell samples truncated to 128
characters. The prompt includes the entity catalog and the exact supported
pattern grammars. Grouping-column and protected-column metadata is sent as
discovery context. NSS then derives replacement columns and all permitted
dependency candidates deterministically from those classifications. The second
pass can only select
contextually useful dependency candidate IDs. Candidates identify edges
selected by the heuristic baseline so that choice remains
available as fallible prior evidence. NSS, rather than the model, derives
group-consistent or record-consistent replacement behavior from the configured
grouping column, excludes protected ordering and timestamp columns, and validates
the assembled plan. Grouping columns remain eligible for replacement so
identifiers such as patient IDs can be anonymized.

Each request permits up to three attempts for transient transport failures or
invalid structured responses. Authentication, authorization, and permanent
configuration failures stop immediately. If structured output remains invalid,
planning fails instead of falling back to the heuristic baseline. Invalid
optional patterns receive up to three focused repair attempts; NSS drops only
the pattern and warns if repair is exhausted.

!!! warning "Inference endpoints receive source data"
    Plan enhancement can send bounded raw cell samples from the full input
    dataframe, including rows that may later be assigned to a holdout set. Enable
    it only when the endpoint is approved to receive the input data.

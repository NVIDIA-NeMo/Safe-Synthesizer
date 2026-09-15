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

## Nemotron Personas sampling

The default `nemotron-personas` sampler draws names, email addresses, phone numbers, and
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
    backend: nemotron-personas
```

To store the files elsewhere, set `replace_pii.sampler.nemotron_personas_path` to
the directory containing `datasets/`, or set `NSS_NEMOTRON_PERSONAS_PATH`:

```yaml
replace_pii:
  sampler:
    backend: nemotron-personas
    nemotron_personas_path: /path/to/managed-assets
```

When an applicable locale asset or required field is unavailable, NSS warns and
uses Faker for that value. Set `backend: faker` to use Faker directly without a
Nemotron Personas dataset. See NVIDIA's
[person-sampling setup](https://docs.nvidia.com/nemo/datadesigner/concepts/person-sampling)
to select a locale resource and the
[`en_US` NGC resource](https://catalog.ngc.nvidia.com/orgs/nvidia/nemotron-personas/resources/nemotron-personas-dataset-en_us/-)
for the example above.

### Data-to-sampler value mappings

Dependency values are matched to sampler values case-insensitively, so values
such as `Female` and `female` need no mapping. By default,
`data_to_sampler_value_mapping` is `auto_discovery`. After the replacement plan is
resolved, NSS compares each dependency column's distinct values with the
selected sampler's values. If unmatched values remain, the configured LLM maps
them to that sampler's vocabulary.

Automatic discovery writes a sparse mapping: identity matches are omitted.
Only unmatched distinct dependency values, each limited to 128 characters,
and the sampler's allowed values are sent to the LLM. If no LLM is configured,
unmatched values produce an error asking for an LLM configuration or a manual
mapping.

For an authoritative manual mapping, replace `auto_discovery` with an inline
mapping keyed by the dependency column names from this dataset:

```yaml
replace_pii:
  sampler:
    backend: nemotron-personas
    data_to_sampler_value_mapping:
      sex:
        Non-binary: null
      race:
        Asian:
          - east asian
          - south asian
          - southeast asian
        Black or African American:
          - black
```

Each nonempty list selects the union of sampler rows with those values. An
explicit `null` disables that condition for the matching input value. Omitted
values continue to use case-insensitive identity matching. Manual mappings are
validated against the resolved dependency columns and the selected sampler's
known values. Faker applies mappings for attributes it supports, such as
gender; unsupported attributes are ignored.

Mappings are sampler-specific. NSS therefore does not accept a separate mapping
file or combine automatic discovery with manual overrides. Generate a resolved
configuration, edit its inline mapping if needed, and run that configuration
again.

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

Resolve and save the plan and sampler-specific dependency mappings from the
full input dataframe without running holdout, model metadata, replacement,
training, generation, or evaluation:

```bash
safe-synthesizer run replace-pii --plan-only \
  --config config.yaml \
  --data-source data.csv
```

The command writes `pii_replacement_config.yaml` in the standard timestamped
NSS run directory under `--artifact-path`. This is a complete NSS configuration
with the resolved plan and inline dependency mappings, so it can be edited and
passed directly to a later run with `--config`.

The matching SDK interface returns the resolved `ReplacePiiConfig` and writes
the complete reusable NSS configuration only when an output path is supplied:

```python
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

config = SafeSynthesizerParameters.from_yaml("config.yaml")
resolved_pii = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .plan_pii_replacement("pii_replacement_config.yaml")
)
```

The generated configuration can be reviewed, edited, and reused directly.

## LLM-assisted planning

The `llm` mapping configures the OpenAI-compatible inference service used for
automatic plan enhancement and data-to-sampler value mapping discovery.

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
Plain HTTP is accepted only for loopback addresses (`localhost`, `127.0.0.0/8`,
or `::1`); any other host must use HTTPS.

The endpoint resolves from the explicit CLI runtime flag, then
`NSS_INFERENCE_ENDPOINT`, then the NSS default; it is never persisted in NSS
configuration. The model resolves from the explicit CLI runtime flag, then
`replace_pii.llm.model_id`, `NSS_INFERENCE_MODEL`, and finally the NSS default.
`NSS_INFERENCE_MODEL` supplies the model only when the configuration omits
`model_id`; use `--inference-model-id` to override a persisted model for one run.
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
    model_id: fastino/gliner2-privacy-filter-PII-multi
    entity_thresholds:
      full_name: 0.95
      first_name: 0.9
      middle_name: 0.9
      last_name: 0.9
      phone_number: 0.5
      date_of_birth: 0.5
      street_address: 0.5
      ssn: 0.5
      national_id: 0.5
      api_key: 0.5
    batch_size: 8
    chunk_length: 384
    chunk_overlap: 128
```

Automatic discovery uses two LLM passes. The first classifies every column's
semantic entity type and may propose a replacement pattern, in bounded batches
of at most 32 profiles and 48 KiB of profile evidence. Each profile contains
deterministic statistics and up to eight distinct cell samples truncated to 128
characters. The prompt includes the entity catalog and the exact supported
pattern grammars. The grouping column is sent as discovery context; protected
columns are not mentioned, because NSS excludes them itself. NSS then derives replacement columns and all permitted
dependency candidates deterministically from those classifications. The second
pass sees only column names, entity types, and proposed patterns, not cell
values, and is split into batches under the same 32-entry and 48 KiB limits. For
each replacement column and each permitted source entity type, it
chooses at most one candidate column, the one describing the same person or
record, or none. The response schema lists only the candidate columns, so the
model cannot name another column or choose two sources of one type, such as
both a person's and a spouse's gender. Each option carries the heuristic
baseline's choice as fallible prior evidence. NSS, rather than the model, derives
group-consistent or record-consistent replacement behavior from the configured
grouping column, excludes protected ordering and timestamp columns, and validates
the assembled plan. Grouping columns remain eligible for replacement so
identifiers such as patient IDs can be anonymized.

Each prompt describes the expected JSON response, and the request also
constrains decoding to that response's strict schema. Each request permits up to
three attempts for transient transport failures or invalid structured responses. Transient failures wait before retrying: the
server's `Retry-After` delay when supplied, otherwise an exponentially growing,
jittered delay, capped at 30 seconds. Authentication, authorization, and
permanent configuration failures stop immediately. If structured output remains invalid,
planning fails instead of falling back to the heuristic baseline. A proposed
pattern that a column cannot use (on an unclassified or protected column, on an
entity type without a pattern syntax, or blank) is ignored rather than rejected.
A pattern must cover at least 99% of a column's non-null values; one that follows
its grammar but covers fewer is dropped with a warning and no repair request,
since a column with mixed formats has no single pattern. Patterns that break their
grammar receive up to three focused repair attempts; NSS drops only the pattern
and warns if every repair response is invalid. Transient failures that persist
through all repair attempts still fail planning, as for any other request. Retry
feedback names the specific columns or dependency conflicts that made the
previous response invalid.

!!! warning "Inference endpoints receive source data"
    Plan enhancement can send bounded raw cell samples from the full input
    dataframe, including rows that may later be assigned to a holdout set. Enable
    it only when the endpoint is approved to receive the input data.

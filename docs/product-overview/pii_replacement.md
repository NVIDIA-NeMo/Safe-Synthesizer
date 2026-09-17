<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

PII replacement v3 uses a dataset-specific replacement plan. The plan names
the columns NSS should replace, the entity type in each column, optional format
patterns, and dependencies between related columns.

NSS applies structured-column and free-text replacements before training. It
resolves plans and sampler mappings against the full input, creates the
holdout, and replaces only the training split so evaluation retains the
original training and test data.

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
such as `Female` and `female` need no mapping. When `replacement_plan` is
`auto_discovery`, NSS discovers `columns_to_replace` and
`dependency_value_mappings` together. It compares each dependency column's
distinct values with the selected sampler's labels. If unmatched values remain,
the configured LLM maps them to that sampler's vocabulary.

Automatic discovery writes a sparse mapping: identity matches are omitted.
Only unmatched distinct dependency values, each limited to 128 characters,
and the sampler's allowed labels are sent to the LLM. If no LLM is configured,
unmatched values produce an error asking for an LLM configuration or a manual
mapping.

For an authoritative plan, place the mapping beside `columns_to_replace`, keyed
by dependency column names from this dataset:

```yaml
replace_pii:
  replacement_plan:
    columns_to_replace:
      - column_name: first_name
        entity_type: first_name
        depends_on:
          - column_name: sex
            entity_type: gender
          - column_name: race
            entity_type: ethnic_background
    dependency_value_mappings:
      sex:
        Non-binary: null
      race:
        Asian:
          - east asian
          - south asian
          - southeast asian
        Black or African American:
          - black
  sampler:
    backend: managed
```

Each nonempty list selects the union of sampler rows with those labels. An
explicit `null` disables that condition for the matching input value. Omitted
values continue to use case-insensitive identity matching. Manual mappings are
validated against the resolved dependency columns and the selected sampler's
known labels. Faker applies mappings for attributes it supports, such as
gender; unsupported attributes are ignored.

Mappings are sampler-specific even though they live in the dataset-specific
replacement plan. NSS does not accept a separate mapping file or combine
automatic discovery with manual overrides. Generate a resolved configuration,
edit its inline plan if needed, and run that configuration again. An explicit
inline or file-based plan never runs a separate mapping-discovery pass; an
omitted mapping is equivalent to `{}`.

## Replacement plan sources

The `replace_pii` configuration has an integer `schema_version`. This release
accepts version `3`; an omitted version is interpreted as version `3`. NSS
includes the version whenever it serializes the configuration.

`replace_pii.replacement_plan` accepts three forms.

### Automatic discovery

Use `auto_discovery` to discover both replacement columns and dependency value
mappings. When `llm` is configured, NSS passes the heuristic result to the LLM
plan enhancer before validating the final plan and mapping unmatched dependency
values.

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
    dependency_value_mappings: {}
```

### Plan file

A plan file is a separately versioned YAML document containing
`schema_version` followed by the same fields as an inline plan:

```yaml
schema_version: 3
columns_to_replace:
  - column_name: email
    entity_type: email
dependency_value_mappings: {}
```

Set `replacement_plan` to its path:

```yaml
replace_pii:
  schema_version: 3
  replacement_plan: ./pii_replacement_plan.yaml
```

Inline plans and plan files are authoritative: NSS validates them against the
input dataframe but does not run heuristic or LLM discovery.

Date-of-birth patterns use Python `strptime`/`strftime` syntax. Named-month
formats such as `%B %d, %Y` match values like `December 10, 1815`; patterns do
not perform general natural-language date parsing, so prose and ordinal forms
such as `December tenth` or `December 10th` are not supported automatically.

## Plan-only workflow

Resolve and save the replacement plan—including its sampler-specific dependency
mappings—from the full input dataframe without running holdout, model metadata,
replacement, training, generation, or evaluation:

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

## Standalone replacement

Apply replacement to a complete input dataframe without running holdout,
training, generation, or evaluation:

```bash
safe-synthesizer run replace-pii \
  --config config.yaml \
  --data-source data.csv \
  --output-file pii_replaced.csv
```

When `--output-file` is omitted, the command adds `_pii_replaced` to the input
stem and writes the result beside `pii_replacement_config.yaml` in the run
directory. For example, `data.csv` produces `data_pii_replaced.csv`. The
resolved configuration is always written so the exact plan and sampler
mappings can be reviewed and reused. Replacement maps and raw detector results
are not persisted.

The SDK equivalent returns the complete `TransformResult` and writes files only
for paths supplied by the caller:

```python
result = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .replace_pii(
        "pii_replaced.csv",
        config_output_path="pii_replacement_config.yaml",
    )
)
```

## Free-text detection

Columns planned as `free_text` use both the configurable GLiNER2 detector and
built-in, structurally validated regex detectors for email addresses, payment
cards, IPv4 addresses, and IPv6 addresses. The default model is
`fastino/gliner2-privacy-filter-PII-multi` and loads lazily only when the plan
contains a free-text target.

```yaml
replace_pii:
  free_text_detection:
    model_id: fastino/gliner2-privacy-filter-PII-multi
    entity_thresholds:
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

Only accepted detector spans are replaced. NSS does not search for other
occurrences, propagate structured values into text, or replace undetected
aliases. Overlapping detections are resolved deterministically, and repeated
accepted values reuse replacements within their configured record or group
scope.

Detector ownership is disjoint. Structurally validated regex exclusively
detects `email`, `credit_debit_card`, `ipv4`, and `ipv6`; NSS does not request
those labels from GLiNER2. GLiNER2 detects the remaining semantic and
contextual entity types except `full_name`. Complete names are detected through
the specific `first_name`, `middle_name`, and `last_name` labels; NSS does not
request the checkpoint's overly broad `person` label. This avoids duplicate
model work and prevents generic person or deterministic-format spans from
competing with more specific detections.

The checkpoint is optimized for recall and can confuse common or domain terms
with names. NSS therefore configures a confidence threshold for every
GLiNER2-detected entity type, with a precision-first `0.9` default for all name
types and `0.5` for the remaining model-detected types. Regex-owned types have
no confidence threshold, and `full_name` has no direct model threshold because
it is not requested. Checkpoint aliases use the threshold of the NSS entity
they normalize to. Lower name thresholds only after calibrating against
representative domain text.

NSS automatically uses CUDA when available and otherwise runs GLiNER2 on CPU.
It deduplicates identical cell text, flattens overlapping chunks across unique
values, and passes `batch_size` to GLiNER2's batched inference API. For CUDA
out-of-memory errors, reduce `batch_size` before changing chunk geometry. The
runtime logs model loading and inference progress without including source
text. See [Program Runtime -- GLiNER2 Device and Throughput](../user-guide/troubleshooting.md#gliner2-device-and-throughput)
for tuning and language-coverage guidance.

Complete birth dates written in natural language, such as `5 April 1990` or
`April 5th, 1990`, are parsed strictly, shifted by the same deterministic
plus-or-minus 365-day policy as structured dates, and rendered as ISO dates
when no explicit strftime pattern exists. Incomplete or vague candidates such
as `spring` are ignored and reported only as an aggregate warning without the
source text.

An independently detected name or street component may reuse a structured
parent replacement only when NSS can align it semantically. Name alignment
requires explicit name-pattern placeholders; street alignment requires an
exact suffix built from typed city, state, postal-code, or country
dependencies. NSS never guesses components from arbitrary token positions or
isolated house numbers.

## LLM-assisted planning

The `llm` mapping configures the OpenAI-compatible inference service used for
automatic plan enhancement and dependency-value mapping discovery.

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
    model_id: fastino/gliner2-privacy-filter-PII-multi
    entity_thresholds:
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

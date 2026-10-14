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

## Nemotron-Personas sampling

The default `nemotron-personas` sampler draws names, email addresses, phone numbers, and
street-address components from the extended Nemotron-Personas locale datasets.
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

When the Nemotron-Personas sampler is selected (the default), its dataset for
the configured locale is required: if the file is missing or cannot be read,
replacement fails instead of switching to Faker. If the plan conditions on a
value the dataset has no column for, such as ethnic background, replacement
also fails. When a persona row lacks one generated field, or the entity type is
not a persona field, NSS uses Faker for that value. Set `backend: faker` to use
Faker without a Nemotron-Personas dataset. See NVIDIA's
[person-sampling setup](https://docs.nvidia.com/nemo/datadesigner/concepts/person-sampling)
to select a locale resource and the
[`en_US` NGC resource](https://catalog.ngc.nvidia.com/orgs/nvidia/nemotron-personas/resources/nemotron-personas-dataset-en_us/-)
for the example above.

### Data-to-sampler value mappings

Dependency values are matched to sampler values case-insensitively, so values
such as `Female` and `female` need no mapping. NSS also maps a few common
gender spellings for every sampler: `F`, `Woman`, and `Girl` to `female`, and
`M`, `Man`, and `Boy` to `male`. These entries are written into the resolved
plan's `data_to_sampler_value_mapping`, where you can edit them; an entry you
write for the same value takes precedence. Any other value needs a mapping. When `replacement_plan` is
`auto_discovery`, NSS discovers `columns_to_replace` and
`data_to_sampler_value_mapping` together. It compares each dependency column's
distinct values with the selected sampler's values. If unmatched values remain,
the configured LLM maps them to that sampler's vocabulary.

Automatic discovery writes a sparse mapping: identity matches are omitted.
Only unmatched distinct dependency values, each limited to 128 characters,
and the sampler's allowed values are sent to the LLM. If no LLM is configured,
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
    data_to_sampler_value_mapping:
      sex:
        Non-binary:
          - female
          - male
      race:
        Asian:
          - east asian
          - south asian
          - southeast asian
        Black or African American:
          - black
  sampler:
    backend: nemotron-personas
```

Each list selects the union of sampler rows with those values, so listing every
sampler value allows any match. Values without an entry use case-insensitive
identity matching. A value that has no entry and matches no sampler value stops
plan resolution with an error naming its column. Non-string dataset values use
their usual text form as keys, for example `1.0` or `2024-01-01T00:00:00`.
Manual mappings are
validated against the resolved dependency columns and the selected sampler's
known values. Faker applies mappings for attributes it supports, such as
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
    data_to_sampler_value_mapping: {}
```

### Plan file

A plan file is a separately versioned YAML document containing
`schema_version` followed by the same fields as an inline plan:

```yaml
schema_version: 3
columns_to_replace:
  - column_name: email
    entity_type: email
data_to_sampler_value_mapping: {}
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
Birth dates are shifted by up to a year. A value that does not match the
pattern is parsed with common date formats instead; if none fits, NSS writes a
random birth date in the pattern's format and logs a warning, so the original
never remains.

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

## LLM-assisted planning

The `llm` mapping configures the OpenAI-compatible inference service used for
automatic plan enhancement and data-to-sampler value mapping discovery.

```yaml
replace_pii:
  schema_version: 3
  replacement_plan: auto_discovery
  llm:
    max_workers: 8
```

An empty mapping (`llm: {}`) enables the NSS inference defaults: NSS runs
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` in a local vLLM server for
the duration of plan discovery; see [Local inference server](#local-inference-server).

### Where planning requests go

Plan discovery sends column statistics and cell samples to an LLM. Where they
go depends only on `NSS_INFERENCE_ENDPOINT` (or `--inference-endpoint-url`):

- Unset: NSS starts its own vLLM server on this machine, and the data stays
  here. This is the default.
- A loopback URL (`localhost`, `127.0.0.0/8`, or `::1`): the data stays on
  this machine. Without a local profile, NSS calls the server you run at that
  address; with `NSS_INFERENCE_LOCAL_PROFILE`, NSS starts its own server there.
- Any other URL: NSS sends the data to that service over HTTPS, for example
  `https://integrate.api.nvidia.com/v1` for the hosted NVIDIA service. Only
  enable it when that service is approved to receive the input data.

The run log states which case applies when planning starts.

With an explicit endpoint and no local profile, NSS never starts a local server,
and the model has no default: it resolves from `--inference-model-id`, then
`replace_pii.llm.model_id`, then `NSS_INFERENCE_MODEL`. `NSS_INFERENCE_MODEL`
supplies the model only when the configuration omits `model_id`; use
`--inference-model-id` to override a persisted model for one run. The endpoint
is never persisted in NSS configuration. Plain HTTP is accepted only for
loopback addresses; any other host must use HTTPS.

Supply an API key at runtime through `NSS_INFERENCE_KEY` or the
`--inference-api-key` CLI option; the hosted NVIDIA service requires one, and
local OpenAI-compatible servers may be keyless. NSS does not store the key in
configuration or plan artifacts.

### Local inference server

When no inference endpoint is set, NSS starts a vLLM server on the local GPU for
plan discovery. The model ID (`replace_pii.llm.model_id`,
`--inference-model-id`, or `NSS_INFERENCE_MODEL`) selects the bundled profile
to run, and defaults to `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`:

```bash
safe-synthesizer run replace-pii --plan-only \
  --config config.yaml \
  --data-source data.csv
```

If vLLM or a CUDA GPU is unavailable, or the GPU is smaller than the profile's
`min_gpu_memory_gib`, preflight and planning fail with an error before
downloading any weights, rather than sending data to a remote service; set
`NSS_INFERENCE_ENDPOINT` to choose one. `NSS_INFERENCE_KEY` is ignored while the
local server runs, and NSS logs a warning if it is set.

Bundled profiles pin each model to a fixed revision, turn on reasoning before
the schema-constrained answer, and fit one 80 GB GPU, such as an A100 or H100;
they require at least 75 GiB, since an 80 GB GPU reports about 79 GiB.
vLLM reserves a fixed share of the GPU's memory for the server, set by the
profile's `gpu_memory_utilization`, regardless of model size; the reservation
holds the weights and the context cache:

| Model ID | GPU memory reserved (80 GB GPU) | Request timeout | Notes |
|----------|---------------------------------|-----------------|-------|
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (default) | about 76 GB (95%); weights about 66 GB | 300 s | Hybrid Mamba mixture of experts; 2,000-token thinking budget |
| `openai/gpt-oss-120b` | about 76 GB (95%); weights about 65 GB | 120 s | Mixture of experts; default medium reasoning effort; fastest |
| `Qwen/Qwen3.8-27B` | about 74 GB (92%); weights about 56 GB | 300 s | Dense; low reasoning effort, 1,000-token thinking budget; slowest |

Nemotron 3.5 Lightning and GPT-OSS-120B both performed well in our PII planning
evaluation. Qwen3.8-27B was less consistent: it did well on some datasets but
scored lower on held-out data, sometimes missed free-text columns, and took
several times longer per table.

Reasoning makes requests slow, so each profile sets its own request timeout.
Set `NSS_INFERENCE_TIMEOUT` or `--inference-timeout-seconds` to override it, for
example for much larger tables. The same setting applies to explicit endpoints,
where it defaults to 60 seconds.

NSS sends no sampling settings of its own, so the server's defaults apply; vLLM
takes them from the model's `generation_config.json`, which for the bundled
models matches their model cards. Each profile also states its sampling
settings explicitly in `request_options`, and the Qwen and Nemotron profiles
add a `thinking_token_budget`, which caps reasoning before the JSON answer. Set
`NSS_INFERENCE_REQUEST_OPTIONS` or `--inference-request-options` to a JSON
object, such as `{"thinking_token_budget": 500}`, to override individual fields
of a profile's request options, or to send fields to an explicit endpoint.
Fields you don't set keep their values; set a field to `null` to drop it.

The first run downloads the weights into the Hugging Face cache; see
[Hugging Face cache and offline](../user-guide/environment.md#hugging-face-cache-and-offline)
to choose where it lives. A model ID without a bundled profile is an error. To
run another model locally, point `NSS_INFERENCE_LOCAL_PROFILE` or the
`--inference-local-profile` CLI option at your own profile YAML (a bundled
profile name such as `gpt-oss-120b` also works); any configured model ID must
then match its served model name, which defaults to its `model_id`. A profile
YAML has these fields: `model_id`, `revision`, and optionally
`served_model_name`, `gpu_memory_utilization`, `max_model_len`,
`max_num_seqs`, `tensor_parallel_size`, `min_gpu_memory_gib` (checked on each
of the `tensor_parallel_size` GPUs; choose GPUs with `CUDA_VISIBLE_DEVICES` in
the shell, not in the profile), `extra_args`, `request_options`,
`environment` (server process variables), `request_timeout_seconds`,
`startup_timeout_seconds`, and `shutdown_timeout_seconds`. In vLLM 0.27,
`thinking_token_budget` needs `VLLM_USE_V2_MODEL_RUNNER: "0"` in `environment`.
Avoid `extra_args` that make vLLM compile kernels at startup, such as
`--mamba-backend=flashinfer`: compiling needs `ninja` and a CUDA compiler,
which an NSS install doesn't include, and it lengthens every server launch.

NSS starts the server only when LLM-assisted discovery runs, meaning an
`auto_discovery` plan with `llm` configured, and stops it before planning
returns, including when planning fails. Plan validation, replacement, and
training therefore never run while the server holds the GPU. Each run loads
the model again; to reuse one server across many runs, start `vllm serve`
yourself and set `NSS_INFERENCE_ENDPOINT` and `NSS_INFERENCE_MODEL` without a
local profile.

#### How the local server is isolated

- Reachable only from this machine: the server listens on a loopback
  address: a free `127.0.0.1` port by default, or the loopback address in
  `NSS_INFERENCE_ENDPOINT` with a local profile. NSS rejects any other address,
  and a port already in use, before launching.
- Usable only by this NSS run: each launch generates a random API key and
  gives it only to the server (through its environment, not its command line)
  and to NSS's own client, so other processes on the machine cannot send it
  requests. `NSS_INFERENCE_KEY` is not used.
- No proxies or request logs: requests to loopback addresses ignore
  `HTTP_PROXY`-style variables, and vLLM request logging stays off because
  prompts contain raw cell samples. Server output goes to the NSS log at debug
  level.
- Short-lived: the server stops when planning ends, and receives SIGTERM if
  the NSS process dies without cleaning up.

### How discovery uses the LLM

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
baseline's choice as fallible prior evidence. NSS, rather than the model, decides how
replacements stay consistent: identifier-like values (email, phone number, SSN,
national ID, card number, API key, IP address, unique identifier) get one
replacement per distinct value across the whole table, and other values are
consistent within a group when a grouping column is configured, else within a
row. Identifier replacements never reuse an original value. Without a pattern,
digit-only identifiers are replaced with numbers up to the column's longest
length, gaining a digit when that range is too full; with a pattern, the
pattern must leave an unused value for every distinct original. Automatic
discovery skips row counters, such as a saved `Unnamed: 0` index column. NSS also excludes protected ordering and timestamp columns, and validates
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

### Free-text detection

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

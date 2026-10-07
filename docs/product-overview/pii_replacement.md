<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

PII replacement v3 uses a dataset-specific replacement plan. The plan names
the columns NSS should replace, the entity type in each column, optional format
patterns, and dependencies between related columns.

Set `replace_pii: null`, pass `--no-replace-pii`, or call
`.with_replace_pii(enable=False)` to run the synthesis pipeline without
replacement.

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
input dataframe but does not run heuristic or LLM discovery for the plan. This
bypass applies only to plan discovery. If `llm` is configured, the replacement
executor can still use it to replace PII found inside free-text columns named by
the plan.

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
    max_workers: 8
```

An empty mapping (`llm: {}`) enables the NSS inference defaults. By default,
NSS runs `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` in a local vLLM server for the duration of plan
discovery; see [Local inference server](#local-inference-server). To use
another OpenAI-compatible service instead, set its endpoint at runtime through
`NSS_INFERENCE_ENDPOINT` or the `--inference-endpoint-url` CLI option. For
example, use `https://integrate.api.nvidia.com/v1` with an API key for the
hosted NVIDIA service, or `http://localhost:8000/v1` for a vLLM server you run
yourself. Plain HTTP is accepted only for loopback addresses (`localhost`,
`127.0.0.0/8`, or `::1`); any other host must use HTTPS.

The endpoint resolves from the explicit CLI runtime flag, then
`NSS_INFERENCE_ENDPOINT`; it is never persisted in NSS configuration. With an
explicit endpoint, the model has no default and must be set; it resolves from
the explicit CLI runtime flag, then `replace_pii.llm.model_id`, then
`NSS_INFERENCE_MODEL`. `NSS_INFERENCE_MODEL` supplies the model only when the
configuration omits `model_id`; use `--inference-model-id` to override a
persisted model for one run. The hosted NVIDIA endpoint requires an
API key. Keyless operation is supported for local OpenAI-compatible endpoints.

Supply the inference API key at runtime through `NSS_INFERENCE_KEY` or the
`--inference-api-key` CLI option. NSS does not store the key in configuration or
plan artifacts.

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

If vLLM or a CUDA GPU is unavailable, planning fails with an error rather than
sending data to a remote service; set `NSS_INFERENCE_ENDPOINT` to choose one.

Bundled profiles pin each model to a fixed revision, turn on reasoning before
the schema-constrained answer, and fit one 80 GB GPU, such as an A100 or H100:

| Model ID | Weights | Request timeout | Notes |
|----------|---------|-----------------|-------|
| `openai/gpt-oss-120b` | about 65 GB | 120 s | Mixture of experts; default medium reasoning effort |
| `Qwen/Qwen3.8-27B` | about 56 GB | 300 s | Dense; low reasoning effort, 1,000-token thinking budget |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (default) | about 66 GB | 300 s | Hybrid Mamba mixture of experts; 2,000-token thinking budget |

Nemotron 3.5 Lightning and GPT-OSS-120B both performed well in our PII planning
evaluation.

Reasoning makes requests slow, so each profile sets its own request timeout.
Set `NSS_INFERENCE_TIMEOUT` or `--inference-timeout-seconds` to override it, for
example for much larger tables. The same setting applies to explicit endpoints,
where it defaults to 60 seconds.

Each profile also sends its model card's recommended sampling settings
instead of the temperature 0 that NSS otherwise sends, since the cards warn
against greedy decoding with reasoning. The Qwen and Nemotron profiles add a
`thinking_token_budget`, which caps reasoning before the JSON answer. Set `NSS_INFERENCE_REQUEST_OPTIONS` or
`--inference-request-options` to a JSON object, such as
`{"temperature": 1.0, "thinking_token_budget": 1000}`, to replace a profile's
request options or the default `{"temperature": 0}` for an explicit endpoint.

The first run downloads the weights into the Hugging Face cache. A model ID
without a bundled profile is an error. To run another model locally, point
`NSS_INFERENCE_LOCAL_PROFILE` or the `--inference-local-profile` CLI option at
your own profile YAML; any configured model ID must then match its served
model name. A profile YAML has these fields: `model_id`,
`revision`, and optionally `served_model_name`, `gpu_memory_utilization`,
`max_model_len`, `max_num_seqs`, `tensor_parallel_size`, `extra_args`,
`request_options`, `environment` (server process variables), `request_timeout_seconds`,
`startup_timeout_seconds`, and `shutdown_timeout_seconds`. In vLLM 0.27,
`thinking_token_budget` needs `VLLM_USE_V2_MODEL_RUNNER: "0"` in `environment`.

NSS starts the server only when LLM-assisted discovery runs, meaning an
`auto_discovery` plan with `llm` configured, and stops it before planning
returns, including when planning fails. Plan validation, replacement, and
training therefore never run while the server holds the GPU. Each run loads
the model again; to reuse one server across many runs, start `vllm serve`
yourself and set `NSS_INFERENCE_ENDPOINT` and `NSS_INFERENCE_MODEL` without a
local profile.

The server listens on a free loopback port. With an explicit local profile, a
loopback `NSS_INFERENCE_ENDPOINT` such as `http://127.0.0.1:8000/v1` selects the
listening address instead; a non-loopback endpoint, or a port already in use,
is an error. Each launch generates its own API key, so `NSS_INFERENCE_KEY` is
ignored. A profile's served model name defaults to its `model_id`.
Server output goes to the NSS log at debug level, and vLLM request logging stays
off because prompts contain raw cell samples.

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
    dataframe, including rows that may later be assigned to a holdout set.
    Free-text replacement can send raw cell values. Enable these operations
    only when the endpoint is approved to receive the input data.

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

Personally identifiable information (PII) is information that can identify a
person, such as a name, email address, phone number, or ID number. Before
training, NSS replaces PII in the input data with realistic fake values, so the
model does not learn the real ones. Replacement follows a dataset-specific
replacement plan. The plan names the columns NSS should replace, the entity
type in each column, optional format patterns, and dependencies between related
columns.

NSS applies structured-column and free-text replacements before training. It
resolves plans and sampler mappings against the full input, creates the
holdout, and replaces only the training split so evaluation retains the
original training and test data.

Set `replace_pii: null`, pass `--no-replace-pii`, or call
`.with_replace_pii(enable=False)` to run the synthesis pipeline without
replacement.

## Nemotron-Personas sampling

Nemotron-Personas are NVIDIA datasets of synthetic people, one per locale,
whose names, sex, ethnic background, and addresses follow that locale's census
distributions. The default `nemotron-personas` sampler draws replacement names
and, for `en_US`, street-address components from them, so names look like the
locale's real population and can match the row's gender and ethnic background
(see [Dependencies](#dependencies)). Email addresses and phone numbers come
from Faker; an email that depends on name columns is built from the replaced
names.

Downloading a dataset requires an NGC API key and the NGC CLI. Data Designer's
[person-sampling setup](https://docs.nvidia.com/nemo/datadesigner/concepts/person-sampling)
explains how to get both, lists the available locales, and describes the same
download steps. NSS reads datasets from the same directory as Data Designer, so
a dataset already downloaded for Data Designer needs no further setup.

Choose the NGC resource matching `replace_pii.locale`. Locale
resources are versioned independently. NSS was tested with version `0.0.2` of
the [`en_US` resource](https://catalog.ngc.nvidia.com/orgs/nvidia/nemotron-personas/resources/nemotron-personas-dataset-en_us/-),
and a later version may rename the columns NSS reads, so download a specific
version rather than the latest:

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
  locale: en_US
```

To store the files elsewhere, set `NSS_NEMOTRON_PERSONAS_PATH` to the directory
containing `datasets/`, or set `replace_pii.sampler.nemotron_personas_path`. A
relative path is resolved against the directory of the configuration file:

```yaml
replace_pii:
  locale: en_US
  sampler:
    backend: nemotron-personas
    nemotron_personas_path: /path/to/managed-assets
```

When `replace_pii.sampler` is not set, NSS uses Nemotron-Personas if the
dataset for the configured locale is present. Otherwise it uses Faker, logs a
warning, and records `backend: faker` in the configuration it writes, unless
the plan maps ethnicity values, which only Nemotron-Personas has; then
resolution fails. A `sampler` you set is always used: if it names
Nemotron-Personas and the file is missing or cannot be read, replacement fails
instead of switching to Faker. If the plan conditions on a value the dataset
has no column for, such as ethnic background, replacement also fails. When a
persona row lacks one generated field, or the entity type is not a persona
field, NSS uses Faker for that value. Set `backend: faker` to use Faker without
a Nemotron-Personas dataset. Faker generates values in `replace_pii.locale`
too, such as French names for `fr_FR`, but cannot condition on ethnic
background.

### Dependencies

A column's `depends_on` lists other columns of the same record that its
replacement must stay consistent with. A dependency works in one of two ways:

- A column that is not replaced, such as `sex` or `race`, narrows what the
  sampler draws and keeps its original value. A first name that depends on
  `sex` comes from a persona of that sex. Give such a column's `entity_type`,
  `gender` or `ethnic_background`, in `depends_on`. Its values must be ones the
  sampler understands; the next section explains how other values are mapped.
- A column that is replaced passes on its new value. An email that depends on
  `first_name` and `last_name` is built from their replacements, and a first
  name that depends on a full-name column takes the first word of the replaced
  full name.

Only these dependencies are allowed:

- First and middle names: gender, ethnic background, or a full name.
- Last names: ethnic background or a full name.
- Full names: gender and ethnic background.
- Email addresses: first, middle, last, or full name, and organization.

A name that depends on a full name cannot also depend on gender or ethnic
background, and an email depends on either name parts or a full name, not both.
Automatic discovery chooses dependencies for you; in a written plan, list them
under `depends_on`.

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
and the sampler's allowed values are sent to the LLM.

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
```

Each list selects the union of sampler rows with those values, so listing every
sampler value allows any match. Values without an entry use case-insensitive
identity matching. A value that has no entry and matches no sampler value stops
plan resolution with an error naming its column.

Gender values are `female` and `male` for every sampler and locale.
Nemotron-Personas also accepts its locale's own words, such as `femme` and
`homme` in `fr_FR`. Ethnicity values exist only in Nemotron-Personas, in
locales whose dataset has an ethnicity column, such as `en_US`.

A plan does not name its sampler: each run checks the mapping against its own
`replace_pii.sampler` and `replace_pii.locale`. An entry the sampler cannot
use stops plan resolution with one error that lists every such entry and names
the sampler, for example a gender value such as `femme` with Faker, or any
ethnicity mapping with Faker. If you switch samplers or locales, rerun automatic
discovery or edit those entries. NSS does not accept a separate mapping file or combine automatic
discovery with manual overrides. Generate a resolved configuration, edit its
plan if needed, and run that configuration again. An inline plan never runs a
separate mapping-discovery pass; an omitted mapping is equivalent to `{}`.

## Replacement settings

The `replace_pii` configuration has an integer `schema_version`. This release
accepts version `3`; an omitted version is interpreted as version `3`. NSS
includes the version whenever it serializes the configuration.

`replace_pii.replacement_plan` is the part of the configuration to review: the
columns to replace and the data-to-sampler value mapping.
`replace_pii.sampler` chooses where generated names and addresses come from
(see [Nemotron-Personas sampling](#nemotron-personas-sampling)).
`replace_pii.locale` sets the locale of generated values and of the
Nemotron-Personas dataset, and `replace_pii.seed` makes replacement
reproducible; when the seed is unset, NSS uses `PERSON_RANDOM_SEED` or `42`, so
the same input gets the same replacements in every run. `llm` and
`free_text_detection` configure how NSS finds PII. `replacement_plan` accepts
three forms.

### Automatic discovery

Use `auto_discovery`, the default, to discover both replacement columns and
dependency value mappings. An LLM classifies the columns and chooses
dependencies, then NSS validates the final plan and maps unmatched dependency
values. Automatic discovery always uses an LLM, which by default runs on this
machine's GPU (see [LLM-assisted planning](#llm-assisted-planning)). To replace
PII without an LLM, use an inline plan; to skip replacement, set
`replace_pii: null`.

```yaml
replace_pii:
  schema_version: 3
  locale: en_US
  replacement_plan: auto_discovery
```

Automatic discovery maps dependency values to the values of
`replace_pii.sampler`, so set the sampler before discovering the plan.

### Inline plan

An inline plan is written directly under `replacement_plan`:

```yaml
replace_pii:
  schema_version: 3
  sampler:
    backend: faker
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

An inline plan is authoritative: NSS validates it against the input dataframe
but does not run heuristic or LLM discovery.

### Plan file

`replace_pii.replacement_plan` can also be the path of a YAML file holding the
same fields, so the plan can be reviewed and reused on its own. A relative path
is resolved against the directory of the configuration file that names it:

```yaml
replace_pii:
  replacement_plan: pii_replacement_plan.yaml
```

A path given with a command-line flag such as `--replace_pii__replacement_plan`,
or set through the SDK, is resolved against the working directory instead.

```yaml
# pii_replacement_plan.yaml
columns_to_replace:
  - column_name: email
    entity_type: email
data_to_sampler_value_mapping: {}
```

In the SDK, pass the path with
`.with_replace_pii(replacement_plan="pii_replacement_plan.yaml")`. On the
command line, `--replace_pii__replacement_plan` takes `auto_discovery` or a
plan file path.

Date-of-birth patterns use Python `strptime`/`strftime` syntax. Named-month
formats such as `%B %d, %Y` match values like `December 10, 1815`; patterns do
not perform general natural-language date parsing, so prose and ordinal forms
such as `December tenth` or `December 10th` are not supported automatically.
Birth dates are shifted by a small random number of days (typically within a month, never more than 180 days, and never zero). A value that does not match the
pattern is parsed with common date formats instead; if none fits, NSS writes a
random birth date in the pattern's format and logs a warning, so the original
never remains.

## Plan-only workflow

Resolve and save the replacement plan, including its dependency mappings, from
the full input dataframe without running holdout, model metadata, replacement,
training, generation, or evaluation:

```bash
safe-synthesizer run replace-pii --plan-only \
  --config config.yaml \
  --data-source data.csv
```

The command writes two files in the standard timestamped NSS run directory
under `--artifact-path`:

- `pii_replacement_plan.yaml`: the resolved `replace_pii.replacement_plan`.
  This is the file to review and edit.
- `safe-synthesizer-config.yaml`: the complete NSS configuration, which names
  `pii_replacement_plan.yaml` and records the sampler used, so it can be passed
  directly to a later run with `--config`.

The plan file starts with a comment naming the sampler and locale it was
written for. That comment is for the reader only; each run checks the plan
against its own `replace_pii.sampler` and `replace_pii.locale`. A short comment
above `columns_to_replace` explains `pattern` and `depends_on`, and the comment
above `data_to_sampler_value_mapping` notes that the section is safe to ignore
unless a run reports an error about it.

The matching SDK interface returns the resolved `ReplacePiiConfig`. When an
output path is supplied, it writes the complete configuration there and
`pii_replacement_plan.yaml` beside it:

```python
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

config = SafeSynthesizerParameters.from_yaml("config.yaml")
resolved_pii = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .plan_pii_replacement("safe-synthesizer-config.yaml")
)
```

Review and edit `pii_replacement_plan.yaml`, then reuse it with
`.with_replace_pii(replacement_plan="pii_replacement_plan.yaml")` or run the
complete configuration again.

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

Only accepted detector spans are replaced. NSS does not search for other
occurrences, propagate structured values into text, or replace undetected
aliases. Overlapping detections are resolved deterministically, and repeated
accepted values reuse replacements within their configured record or group
scope.

Detector ownership is disjoint. Structurally validated regex exclusively
detects `email`, `credit_debit_card`, `ipv4`, and `ipv6`; NSS does not request
those labels from GLiNER2. GLiNER2 detects the remaining semantic and
contextual entity types, including the checkpoint's specific `full_name` label.
NSS does not request the broader `person` label. This avoids duplicate model
work and prevents generic person or deterministic-format spans from competing
with more specific detections.

The checkpoint is optimized for recall and can confuse common or domain terms
with names. NSS therefore configures a confidence threshold for every
GLiNER2-detected entity type, with a precision-first `0.95` default for
`full_name`, `0.9` for the specific name-part types, and `0.5` for the remaining
model-detected types. Regex-owned types have no confidence threshold. Checkpoint
aliases use the threshold of the NSS entity they normalize to. Lower name
thresholds only after calibrating against representative domain text.

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

An independently detected name or street component may reuse a parent
replacement only when NSS can align it semantically. A full name, whether in a
structured column or detected in free text, aligns with first, middle, and
last names through explicit name-pattern placeholders, or, without a pattern,
when it is a plain two- or three-word name such as `Kenneth Gonzales`. A free-text
full name like that is replaced word by word, so `Kenneth Gonzales` and a bare
`Kenneth` in the same record or group get the same first-name replacement.
Names with titles, initials, or suffixes, such as `Dr. Ada Lovelace` or
`Ada K. Lovelace`, are replaced whole and not split. Street alignment requires
an exact suffix built from typed city, state, postal-code, or country
dependencies. NSS never guesses components from other token positions or
isolated house numbers.

## LLM-assisted planning

Automatic discovery uses an LLM for plan enhancement and data-to-sampler value
mapping discovery. By default, NSS runs
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` in a local vLLM server for
the duration of plan discovery; see [Local inference server](#local-inference-server).
Without a GPU, set an inference endpoint instead (see
[Where planning requests go](#where-planning-requests-go)); otherwise preflight
and planning fail with an error. The optional `llm` mapping changes how requests
are made:

```yaml
replace_pii:
  schema_version: 3
  llm:
    max_workers: 8
```

`llm` cannot be `null`. To replace PII without an LLM, give an inline plan.

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

NSS starts the server only when automatic discovery runs without an inference
endpoint, and stops it before planning
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

Automatic discovery sends four kinds of LLM request, in this order:

1. Entity types. Each column's profile, made of deterministic statistics and up
   to eight distinct cell samples truncated to 128 characters, is sent with the
   entity catalog, and the model classifies the column. The grouping column is
   sent as context; protected columns are not sent, because NSS excludes them
   itself.
2. Patterns. For unprotected columns whose entity type has a pattern syntax,
   the model proposes a replacement pattern from their profiles, entity types,
   and the pattern grammars they use. The request is skipped when no column
   qualifies. Keeping pattern rules out of the type request leaves the model's
   reasoning for the types.
3. Dependencies. NSS derives the replacement columns and every permitted
   dependency candidate from the classifications. This request sees only column
   names, entity types, and proposed patterns, not cell values. For each
   replacement column and each permitted source entity type, the model chooses
   at most one candidate column, the one describing the same person or record,
   or none. The response schema lists only the candidate columns, so the model
   cannot name another column or choose two sources of one type, such as both a
   person's and a spouse's gender. The prompt asks the model to choose a
   candidate whenever it describes the same person or record, so the fake value
   stays consistent with it.
4. Value mappings. After NSS validates the plan, dependency values that match
   no sampler value are sent with the sampler's values in one request; see
   [Data-to-sampler value mappings](#data-to-sampler-value-mappings). When the
   request would exceed its size limit, planning fails and asks for a manual
   mapping.

The first three are split into batches of at most 32 columns and 48 KiB; the
type and pattern requests share batches.

NSS, rather than the model, decides how
replacements stay consistent: within a group when a grouping column is
configured, else within a row. Values that differ only in letter case count as
the same value: `Ada`, `ADA`, and `ada` share one replacement, written in each
occurrence's case, such as `Kathleen`, `KATHLEEN`, and `kathleen`. The same
value of the same type also gets one replacement across columns that share a
format: an address in `email` and `backup_email` of one record is replaced with
the same address, while phones written as `(###) ###-####` in one column and
`###.###.####` in another are replaced separately. Every replacement differs
from the value it replaces, ignoring case. A replaced full name also avoids the original's first and last name;
when no such name is found, NSS keeps one that shares a part and logs a warning
with the count. A replacement may equal another row's original value, and
replacements may repeat across rows. The exception is an identifier-like column
(email, phone number, SSN, national ID, card number, API key, IP address, unique
identifier) whose original values are all distinct, such as a key column: its
replacements stay distinct too, ignoring case, and when no unused value is
found, NSS repeats one and logs a warning with the count. Without a pattern,
digit-only identifiers are replaced with numbers up to the column's longest
length; for a column that must stay distinct, the range gains a digit when it is
too small. Automatic
discovery skips row counters, such as a saved `Unnamed: 0` index column. NSS also excludes protected ordering and timestamp columns, and validates
the assembled plan. Grouping columns remain eligible for replacement so
identifiers such as patient IDs can be anonymized. A replaced grouping column
is one-to-one: no two groups get the same replacement, so their series never
merge. A masked identifier such as `ID-#` gains a position when it has too few
unused values for every group, and the run fails rather than merge two groups
when no distinct value can be found.

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
since a column with mixed formats has no single pattern. An email column without
a pattern that depends on name columns is still built from the replaced names, as
`{first}.{last}##@{domain}` with two random digits and the original domain, or
`{first}##@{domain}` when only one name part is known. In any email, spaces
inside a name become dots, so `De La Cruz` is written as `de.la.cruz`. Patterns that break their
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

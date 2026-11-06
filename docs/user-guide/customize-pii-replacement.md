<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Customizing PII Replacement

This guide shows how to review and edit the PII replacement plan, choose where
synthetic values come from, tune free-text detection, and choose the model that
builds the plan. For what PII replacement does and why, see
[PII Replacement](../product-overview/pii_replacement.md). For a list of every
setting, see [Configuration Reference -- Replacing PII](configuration.md#replacing-pii).

## Turn Replacement Off

Set `replace_pii: null` in YAML, pass `--no-replace-pii` on the command line, or
call `.with_replace_pii(enable=False)` in the SDK.

The PII replay check in the evaluation still needs to know which values are
PII, so evaluation runs PII planning on the training data. Planning uses the
local planning model, or sends samples to `NSS_INFERENCE_ENDPOINT` when it is
set; see [Where planning requests go](environment.md#where-planning-requests-go).
Set `evaluation.pii_replay_enabled: false` to skip it.

## Review and Reuse the Plan

The replacement plan lists the columns to replace, the kind of PII in each, an
optional format pattern, and which columns each replacement must agree with.
Create it without running anything else:

```bash
safe-synthesizer run replace-pii --plan-only \
  --config config.yaml \
  --data-source data.csv
```

The command writes two files to a new timestamped run directory under
`--artifact-path`:

- `pii_replacement_plan.yaml`: the plan. This is the file to review and edit.
- `safe-synthesizer-config.yaml`: a complete configuration that names the plan
  file and records the sampler used. Pass it to a later run with `--config`.

A full pipeline run writes the same two files to its run directory.

In the SDK, `plan_pii_replacement()` returns the resolved `ReplacePiiConfig`.
When you pass an output path, it writes the configuration there and
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

To reuse the reviewed plan, run the saved `safe-synthesizer-config.yaml`
again, since it already names the plan file. To point a different
configuration at the plan, set `replace_pii.replacement_plan` in its YAML, or
in the SDK call `.with_replace_pii(replacement_plan="pii_replacement_plan.yaml")`;
its other `replace_pii` settings, such as `locale`, are kept.

The [Review the PII Replacement Plan](../tutorials/pii-replacement-review.ipynb)
tutorial walks through this end to end.

## Plan Sources

`replace_pii.replacement_plan` takes one of three forms.

### Automatic discovery

`auto_discovery`, the default, builds the plan with an LLM (see
[Planning LLM](#planning-llm)). Row counters, such as a saved `Unnamed: 0`
index column, are skipped, and so are the columns named in
`data.order_training_examples_by` and `time_series.timestamp_column`. The
grouping column can be replaced, so identifiers such as patient IDs are
anonymized.

```yaml
replace_pii:
  replacement_plan: auto_discovery
```

Discovery matches values in your data to the sampler's values (see
[Value mappings](#value-mappings)), so set `replace_pii.sampler` before
discovering the plan.

### Inline plan

Write the plan directly in the configuration. An inline plan makes no LLM
request, so this is how to replace PII without an LLM:

```yaml
replace_pii:
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

Give the path of a YAML file holding the same fields, such as the
`pii_replacement_plan.yaml` from a plan-only run:

```yaml
replace_pii:
  replacement_plan: pii_replacement_plan.yaml
```

A relative path in a configuration file is resolved against that file's
directory. A path given on the command line
(`--replace_pii__replacement_plan`) or through the SDK is resolved against the
working directory.

!!! note "Plans you write are checked less strictly"
    Safe Synthesizer validates every plan against the data, but a plan you
    write is allowed to run where a discovered plan would not. For example, a
    pattern that fits fewer than 99% of a column's values is dropped from a
    discovered plan but kept, with a warning, in a plan you write. Value
    mapping problems stop every plan with an error: a mapping entry the sampler
    cannot use, or a dependency value with no mapping entry and no sampler match.

## Entity Types

Each column in the plan has an `entity_type`. Types marked "kept" are never
replaced, so they cannot appear in `columns_to_replace`. Of those, `gender`,
`ethnic_background`, and `organization` can appear in `depends_on`; the rest
are only recognized by discovery so it does not mistake them for PII.

| Entity type | Replaced | Pattern syntax |
|-------------|----------|----------------|
| `first_name`, `middle_name`, `last_name`, `full_name` | Yes | Name template |
| `email` | Yes | Name template |
| `phone_number` | Yes | Character mask |
| `credit_debit_card`, `api_key`, `unique_identifier` | Yes | Character mask |
| `date_of_birth` | Yes, shifted by days or weeks | Date format |
| `street_address`, `national_id`, `ipv4`, `ipv6` | Yes | None |
| `free_text` | PII inside the text | None |
| `gender`, `ethnic_background`, `organization` | Kept | None |
| `date`, `city`, `state`, `zipcode`, `country` | Kept | None |

## Patterns

A column's optional `pattern` sets the format of its replacement values. It
should match at least 99% of the column's non-null values. Values the pattern
does not fit still get replacements written in the pattern. If `pattern` is
left out, the entity's default format is used; an empty pattern is rejected.

Name templates, for names and `email`, combine literal text with placeholders,
such as `{First} {Last}` or `{f}.{last}@{domain}`:

- `{first}`, `{middle}`, `{last}`: the name part in lowercase. `{First}` writes
  it in title case and `{FIRST}` in uppercase, and likewise for middle and last.
- `{f}`, `{m}`, `{l}`: the part's lowercase initial; `{F}`, `{M}`, `{L}` write
  it in uppercase.
- `{domain}`, `{organization}`: email only. The original email's domain (a synthetic
  one when the original has none), and the value of an organization dependency
  written as a domain label. An email pattern must contain `@`, and `#` writes
  one random digit.

A full name without a pattern keeps the original's professional titles and
suffixes, such as `Dr.` and `Jr.`, and drops forms of address such as `Mr.` and
`Mrs.`. A full name with a pattern is written exactly as the pattern says:
`{First} {Last}` writes no title, and `Dr. {First} {Last}` writes `Dr.` in
every value.

Character masks, for phone numbers, card numbers, API keys, and identifiers,
write one random character per token and copy everything else, so `pmc-######`
produces values like `pmc-482913`:

- `#` is a digit, `^` an uppercase letter, `@` a lowercase letter, `&` a digit
  or uppercase letter, `%` a digit or lowercase letter, and `*` a digit or
  letter.
- `[01]` is one character from the brackets. Brackets list characters, not
  ranges, so `CUST-10[01]###` starts with `CUST-100` or `CUST-101`.
- `\` makes the next token or bracket literal, so `\#` writes `#`. A `]` without
  an opening `[` is rejected; write it as `\]`.

Date formats, for `date_of_birth`, use Python `strftime` codes such as
`%m/%d/%Y` or `%B %d, %Y`. Prose and ordinal forms such as `December 10th` are
not supported.

## Dependencies

A column's `depends_on` lists other columns of the same record that its
replacement must agree with. A dependency works in one of two ways:

- A column that is not replaced, such as `sex` or `race`, steers what the
  sampler draws and keeps its original value. A first name that depends on
  `sex` comes from a persona of that sex. Give such a column's `entity_type`,
  `gender` or `ethnic_background`, in `depends_on`.
- A column that is replaced passes on its new value. An email that depends on
  `first_name` and `last_name` is built from their replacements, and a first
  name that depends on a full-name column takes the first name of the replaced
  full name.

Only these dependencies are allowed:

- First and middle names: gender, ethnic background, or a full name.
- Last names: ethnic background or a full name.
- Full names: gender and ethnic background.
- Email addresses: first, middle, last, or full name, and organization.

A name that depends on a full name cannot also depend on gender or ethnic
background, and an email depends on either name parts or a full name, not both.

## Value Mappings

A dependency column's values must be ones the sampler understands. Matching
ignores case, and common gender spellings (`F`, `Woman`, `Girl`, `M`, `Man`,
`Boy`) are mapped automatically. Any other value needs an entry in
`data_to_sampler_value_mapping`. Automatic discovery fills it in for you; in a
plan you write, add entries keyed by the dependency column's name:

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

Each list allows any of the sampler values it names. Gender values are `female`
and `male` for every sampler; Nemotron-Personas also accepts its locale's own
words, such as `femme` in `fr_FR`. Ethnicity values exist only in
Nemotron-Personas, for locales whose dataset has them, such as `en_US`.

The mapping is checked against each run's sampler and locale. If you switch
either, rerun automatic discovery or edit the entries.

## Samplers

The sampler generates replacement values. Set it with
`replace_pii.sampler.backend`. When you leave it unset, Safe Synthesizer uses
Nemotron-Personas if the dataset for your locale is downloaded, and Faker
otherwise, with a warning. The exception is a discovered plan that matches on
ethnic background, which only Nemotron-Personas supports: it fails instead.

### Nemotron-Personas

Nemotron-Personas are NVIDIA datasets of synthetic people, one per locale,
whose names, sex, ethnic background, and addresses follow that locale's census
distributions. The `nemotron-personas` sampler draws names and, for `en_US`,
street addresses from them, so names look like the locale's real population
and can match the record's gender and ethnic background. Emails and phone
numbers come from Faker.

Downloading a dataset requires an NGC API key and the NGC CLI. Data Designer's
[person-sampling setup](https://docs.nvidia.com/nemo/datadesigner/concepts/person-sampling)
explains how to get both and lists the available locales. Safe Synthesizer reads
datasets from the same directory as Data Designer, so a dataset already
downloaded for Data Designer needs no further setup.

Download a specific version of the resource for your locale. Safe Synthesizer
was tested with version `0.0.2` of the
[`en_US` resource](https://catalog.ngc.nvidia.com/orgs/nvidia/nemotron-personas/resources/nemotron-personas-dataset-en_us/-),
and a later version may rename columns:

```bash
ngc registry resource download-version \
  nvidia/nemotron-personas/nemotron-personas-dataset-en_us:0.0.2
mkdir -p "${HOME}/.data-designer/managed-assets/datasets"
cp nemotron-personas-dataset-*/*.parquet \
  "${HOME}/.data-designer/managed-assets/datasets/"
```

The sampler loads `datasets/<locale>.parquet`, so `replace_pii.locale` must
match the file name, such as `en_US` for `en_US.parquet`. To store the files
elsewhere, set `NSS_NEMOTRON_PERSONAS_PATH` or
`replace_pii.sampler.nemotron_personas_path` to the directory containing
`datasets/`.

If you set `backend: nemotron-personas` and the dataset is missing, the run
fails rather than switching to Faker.

### Faker

Set `backend: faker` to generate every value with
[Faker](https://faker.readthedocs.io/), without downloading a dataset. Faker
supports [many more locales](https://faker.readthedocs.io/en/master/locales.html)
than Nemotron-Personas and applies gender dependencies, but it cannot match
ethnic background.

## Free-Text Detection

Columns planned as `free_text` are scanned with the GLiNER2 model
`fastino/gliner2-privacy-filter-PII-multi` plus regex checks for email
addresses, payment cards, and IP addresses. Only the detected PII is replaced.
GLiNER2 needs a CUDA GPU; see
[GLiNER2 Device and Throughput](troubleshooting.md#gliner2-device-and-throughput)
for memory and speed tuning.

The defaults favor precision: a word is replaced only when the model is very
confident it is PII. The settings you are most likely to change are:

- `entity_thresholds`: minimum confidence per entity type. Lower a value to
  catch more PII of that type, at the cost of replacing more words that are
  not PII.
- `name_exclusion_thresholds`: labels for terms often mistaken for names, such
  as medical conditions and family relationships. A detected name overlapping
  one of these is not replaced, unless it is a value already known in the
  record. Labels you set are added to the defaults or change
  their thresholds; set a label to `null` to drop it. Dropping every default
  label turns exclusion off, for text where missed names matter more than
  false ones.
- `find_and_replace`: when `true` (the default, `en_US` only), a value already
  known in the record, such as the original first name, is also replaced
  wherever it appears in free text, even if the model missed it. Short values
  and common English words, such as `Will` or `June`, are skipped.
- `batch_size`: reduce it if GLiNER2 runs out of GPU memory.

```yaml
replace_pii:
  free_text_detection:
    entity_thresholds:
      first_name: 0.9
    find_and_replace: true
    batch_size: 8
```

Entity types you leave out of `entity_thresholds` keep their defaults. See
[`FreeTextDetectionConfig`][nemo_safe_synthesizer.config.replace_pii.FreeTextDetectionConfig]
for every field and default.

## Locales Other Than en_US

`replace_pii.locale` sets the locale of generated values, but parts of
replacement were built and tuned for English text and US conventions:

- Free-text detection was tuned on English text. The default GLiNER2
  checkpoint covers English, French, Spanish, German, Italian, Portuguese, and
  Dutch, and it needs spaces between words, so most PII in Chinese, Japanese,
  Korean, or Thai sentences is not detected.
- Name handling in free text knows mostly English titles and suffixes. It reads
  French `M.` as a first-name initial and a Spanish second surname, as in
  `Ana García López`, as a middle name.
- Email addresses built from generated names keep accented and non-Latin
  letters, such as `françois.prévost42@example.com`.
- Full street addresses keep their city and postal code only when the street
  comes first; other address orders are replaced whole.
- `find_and_replace` runs only for `en_US`.

## Planning LLM

Automatic discovery uses an LLM to classify columns, propose patterns, choose
dependencies, and map values. By default, Safe Synthesizer starts
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` in a temporary vLLM server on
this machine's GPU and stops it before replacement and training begin. No API
key is needed and the data stays on this machine. The first run downloads
about 66 GB of weights into the Hugging Face cache.

To use a hosted or self-hosted endpoint instead, set `NSS_INFERENCE_ENDPOINT`;
see [Environment Variables -- Where planning requests go](environment.md#where-planning-requests-go).

### Local inference server

Safe Synthesizer can run two planning models locally. Each needs one GPU with
at least 75 GiB of memory, such as an 80 GB A100 or H100. Without vLLM or a
large enough GPU, planning fails with an error before downloading anything,
rather than sending data elsewhere.

Choose a planning model with `replace_pii.llm.model_id`, `--inference-model-id`,
or `NSS_INFERENCE_MODEL`:

| Model ID | Request timeout | Notes |
|----------|-----------------|-------|
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (default) | 300 s | Performed well in our evaluation |
| `openai/gpt-oss-120b` | 120 s | Performed well in our evaluation; fastest |

Override the request timeout for very large tables with `NSS_INFERENCE_TIMEOUT`
or `--inference-timeout-seconds`. Override request fields, such as the thinking
budget, with `NSS_INFERENCE_REQUEST_OPTIONS` or `--inference-request-options`,
for example `{"thinking_token_budget": 500}`.

To run another model locally, point `NSS_INFERENCE_LOCAL_PROFILE` or
`--inference-local-profile` at your own profile YAML. If you also set
`replace_pii.llm.model_id` or `NSS_INFERENCE_MODEL`, it must match the
profile's served model name, which defaults to its `model_id`. A profile has
`model_id` and `revision`, plus optional vLLM settings such as
`gpu_memory_utilization`, `max_model_len`, `tensor_parallel_size`,
`min_gpu_memory_gib`, `extra_args`, `request_options`, `environment`, and the
request, startup, and shutdown timeouts. The built-in profiles in
`src/nemo_safe_synthesizer/pii_replacer/local_inference/profiles/` are good
starting points; if you set `thinking_token_budget`, copy their `environment`
settings too. Avoid `extra_args` that make vLLM compile kernels at startup, such as
`--mamba-backend=flashinfer`: a Safe Synthesizer install has no CUDA compiler.

`replace_pii.llm.max_workers` (default 8) sets how many planning requests run
at once.

### Reuse one server across runs

Each run starts the planning model again, which takes about a minute and a half
even after the first download. When you plan or replace PII many times, for
example while refining a plan, start the server once yourself and point Safe
Synthesizer at it. This command starts the same server as the default profile,
`nemotron-3.5-lightning-30b-a3b.yaml` in
`src/nemo_safe_synthesizer/pii_replacer/local_inference/profiles/`; if the
two ever differ, follow the profile.

```bash title="Start the planning server"
# The V1 model runner keeps the thinking budget working.
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --mamba-ssm-cache-dtype=float16 \
  --reasoning-parser=nemotron_v3 \
  --default-chat-template-kwargs='{"enable_thinking": true}'
```

Run it in the Python environment where Safe Synthesizer is installed, which
includes vLLM. The server is ready when
`curl -s http://127.0.0.1:8000/v1/models` lists the model. Then set these
variables where you run Safe Synthesizer:

```bash title="Use the planning server"
export NSS_INFERENCE_ENDPOINT=http://127.0.0.1:8000/v1
export NSS_INFERENCE_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
export NSS_INFERENCE_TIMEOUT=300
export NSS_INFERENCE_REQUEST_OPTIONS='{"temperature": 1.0, "top_p": 0.95, "thinking_token_budget": 4000}'
```

Keep the last two lines. With an endpoint, Safe Synthesizer applies none of the
profile's settings, so without them each request times out after 60 seconds
and the model reasons without a token budget. In a notebook, set the same
values in `os.environ` before planning starts.

- The server holds 95% of the GPU's memory until you stop it with Ctrl-C, so
  use it for `plan_pii_replacement()`, `replace_pii()`, and
  `safe-synthesizer run replace-pii`. Stop it, and unset
  `NSS_INFERENCE_ENDPOINT`, before training or generating on the same GPU.
- Binding to `127.0.0.1` keeps the data on this machine, but any user of the
  machine can call the server. To require a key, start the server with
  `VLLM_API_KEY` set and set `NSS_INFERENCE_KEY` to the same value.

## Standalone Replacement

Replace PII in a complete dataset without holdout, training, generation, or
evaluation:

!!! note "Use standalone replacement to check the plan"
    Standalone replacement is for reviewing and refining PII replacement
    before a full run, not a privacy step on its own. It replaces only the
    kinds of PII it knows and can miss PII in free text, so its output can
    still identify people. Run the full pipeline to produce data you share.

```bash
safe-synthesizer run replace-pii \
  --config config.yaml \
  --data-source data.csv \
  --output-file pii_replaced.csv
```

The output is Parquet when its path ends in `.parquet`, and CSV otherwise.
Without `--output-file`, the command writes `<input name>_pii_replaced.csv` to
the run directory, or `.parquet` for Parquet input, such as
`data_pii_replaced.csv` for `data.csv`. The plan and configuration files are
always written, as in [Review and Reuse the Plan](#review-and-reuse-the-plan).

The SDK equivalent returns a `TransformResult` and writes only the files you
name:

```python
result = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .replace_pii(
        "pii_replaced.csv",
        config_output_path="safe-synthesizer-config.yaml",
    )
)
```

## How Replacement Works Internally

For the exact rules -- how replacements stay consistent and distinct, how
discovery batches and retries LLM requests, which detector owns which entity
type, and how the local server is isolated -- see
[PII Replacement Internals](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/design.md#pii-replacement-internals)
in `design.md`.

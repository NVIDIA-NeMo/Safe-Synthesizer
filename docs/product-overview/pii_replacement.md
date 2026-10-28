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

This page has three parts:

- [Discovery](#discovery) builds the replacement plan.
- [Replacement](#replacement) applies the plan to the data.
- [Self-hosted LLM](#self-hosted-llm) configures the LLM that discovery can use.

## Discovery

Discovery decides which columns to replace, the entity type and pattern of each,
the dependencies between them, and how dependency values map to the sampler's
values. The result is the replacement plan.

### Replacement settings

The `replace_pii` configuration has an integer `schema_version`. This release
accepts version `3`; an omitted version is interpreted as version `3`. NSS
includes the version whenever it serializes the configuration.

`replace_pii.replacement_plan` is the part of the configuration to review: the
columns to replace and the data-to-sampler value mapping. `replace_pii.sampler`
chooses where generated names and addresses come from (see
[Samplers](#samplers)). `replace_pii.locale` sets the locale of generated values and of the
Nemotron-Personas dataset, and `replace_pii.seed` makes replacement
reproducible; when the seed is unset, NSS uses `PERSON_RANDOM_SEED` or `42`, so
the same input gets the same replacements in every run. `llm` and
`free_text_detection` configure how NSS finds PII. `replacement_plan` accepts
three forms.

#### Automatic discovery

Use `auto_discovery`, the default, to discover both replacement columns and
dependency value mappings. An LLM classifies the columns and chooses
dependencies, then NSS validates the final plan and maps unmatched dependency
values; see [How discovery uses the LLM](#how-discovery-uses-the-llm).
Automatic discovery always uses an LLM, which by default runs on this machine's
GPU (see [Self-hosted LLM](#self-hosted-llm)). To replace PII without an LLM,
use an inline plan; to skip replacement, set `replace_pii: null`.

```yaml
replace_pii:
  schema_version: 3
  locale: en_US
  replacement_plan: auto_discovery
```

Automatic discovery maps dependency values to the values of
`replace_pii.sampler`, so set the sampler before discovering the plan.

Automatic discovery skips row counters, such as a saved `Unnamed: 0` index
column. NSS also excludes protected ordering and timestamp columns, and
validates the assembled plan. Grouping columns remain eligible for replacement
so identifiers such as patient IDs can be anonymized.

#### Inline plan

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

#### Plan file

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
type and pattern requests share batches. The model only proposes the plan; how
replacements stay consistent is decided by NSS, as described under
[Replacement rules](#replacement-rules).

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

Where these requests go, and how to run the model on this machine, is described
under [Self-hosted LLM](#self-hosted-llm).

!!! warning "Inference endpoints receive source data"
    Plan enhancement can send bounded raw cell samples from the full input
    dataframe, including rows that may later be assigned to a holdout set. Enable
    it only when the endpoint is approved to receive the input data.

### Plan-only workflow

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
unless a run reports an error about it. A full pipeline run writes the same two
files to its run directory.

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

## Replacement

Replacement applies the plan: it generates a fake value for every planned
structured cell, and for every PII span detected in free-text cells.

### Samplers

The sampler generates replacement values. Set it with
`replace_pii.sampler.backend`.

#### Nemotron-Personas

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
has no column for, such as ethnic background, replacement also fails. When a persona row lacks one generated field, or the entity type is
not a persona field, NSS uses Faker for that value.

#### Faker

Set `backend: faker` to generate every value with Faker, without a
Nemotron-Personas dataset. Faker uses `replace_pii.locale`, such as French
names for `fr_FR`, and applies gender dependencies; it cannot condition on
ethnic background.

### Replacement rules

NSS, rather than the model or the sampler, decides how replacements stay
consistent: within a group when a grouping column is configured, else within a
row. Values that differ only in letter case count as the same value: `Ada`,
`ADA`, and `ada` share one replacement, written in each occurrence's case, such
as `Kathleen`, `KATHLEEN`, and `kathleen`. The same value of the same type also
gets one replacement across columns that share a format: an address in `email`
and `backup_email` of one record is replaced with the same address, while
phones written as `(###) ###-####` in one column and `###.###.####` in another
are replaced separately.

Every replacement differs from the value it replaces, ignoring case. A replaced
full name also avoids the original's first and last name; when no such name is
found, NSS keeps one that shares a part and logs a warning with the count. A
replacement may equal another row's original value, and replacements may repeat
across rows. The exception is an identifier-like column (email, phone number,
SSN, national ID, card number, API key, IP address, unique identifier) whose
original values are all distinct, such as a key column: its replacements stay
distinct too, ignoring case, and when no unused value is found, NSS repeats one
and logs a warning with the count. Without a pattern, digit-only identifiers are
replaced with numbers up to the column's longest length; for a column that must
stay distinct, the range gains a digit when it is too small.

A replaced grouping column is one-to-one: no two groups get the same
replacement, so their series never merge. A masked identifier such as `ID-#`
gains a position when it has too few unused values for every group, and the run
fails rather than merge two groups when no distinct value can be found.

An email column without a pattern that depends on name columns is still built
from the replaced names, as `{first}.{last}##@{domain}` with two random digits
and the original domain, or `{first}##@{domain}` when only one name part is
known. In any email, spaces inside a name become dots, so `De La Cruz` is
written as `de.la.cruz`.

Birth dates are shifted by a small random number of days (typically within a
month, never more than 180 days, and never zero). A value that does not match
the pattern is parsed with common date formats instead; if none fits, NSS writes
a random birth date in the pattern's format and logs a warning, so the original
never remains.

### Free-text detection

Columns planned as `free_text` use both the configurable GLiNER2 detector and
built-in, structurally validated regex detectors for email addresses, payment
cards, IPv4 addresses, and IPv6 addresses. The default model is
`fastino/gliner2-privacy-filter-PII-multi`, pinned to the Hugging Face commit
the default thresholds were tuned on, and it loads lazily only when the plan
contains a free-text or street-address target. A different `model_id` loads its latest revision
unless `model_revision` is set.

```yaml
replace_pii:
  free_text_detection:
    model_id: fastino/gliner2-privacy-filter-PII-multi
    model_revision: c153999da5f4c509df4322b0c6a1baf3d2c284d7
    entity_thresholds:
      first_name: 0.946
      middle_name: 0.7
      last_name: 0.618
      full_name: 0.964
      phone_number: 0.990
      date_of_birth: 0.9998
      street_address: 0.985
      ssn: 0.050
      national_id: 0.982
      api_key: 0.992
      unique_identifier: 0.947
    name_exclusion_thresholds:
      body_part: 0.062
      family_relationship: 0.062
      gender: 0.385
      medical_condition: 0.393
      organism: 0.051
    require_middle_name_neighbors: true
    find_and_replace: true
    batch_size: 8
    chunk_length: 384
    chunk_overlap: 128
```

When detections overlap, the longer span wins; between spans of equal length,
the more confident one wins, with regex matches counting as fully confident.

Detector ownership is disjoint. Structurally validated regex exclusively
detects `email`, `credit_debit_card`, `ipv4`, and `ipv6`; NSS does not request
those labels from GLiNER2. GLiNER2 detects the remaining semantic and
contextual entity types, including the checkpoint's specific `full_name` label.
NSS does not request the broader `person` label. This avoids duplicate model
work and prevents generic person or deterministic-format spans from competing
with more specific detections.

Several checkpoint labels map to one NSS entity type:

| NSS entity type | GLiNER2 labels requested |
|---|---|
| `phone_number` | `phone_number`, `fax_number` |
| `street_address` | `street_address` |
| `national_id` | `national_id_number`, `passport_number`, `tax_id`, `tax_number`, and `government_id` values that are not SSNs |
| `ssn` | `government_id` values shaped `ddd-dd-dddd` with an assignable area, group, and serial (not area 000, 666, or 900-999, group 00, or serial 0000) |
| `unique_identifier` | `customer_id`, `employee_id`, `medical_record_number`, `member_id`, `health_plan_beneficiary_number`, `certificate_number` |

Identifiers are requested only by the kinds that text names, such as "customer
ID 1234", because generic identifier labels mostly match device, vehicle, and
biometric codes. A detected identifier without a digit is dropped. Certificate
and professional license numbers are identifiers. Tax numbers are national IDs,
including business tax IDs, which the checkpoint does not distinguish from
personal ones. Driver's license labels are not requested because they mostly
match license plates and other vehicle codes.

The checkpoint is optimized for recall and can confuse common or domain terms
with names. NSS therefore configures a confidence threshold for every
GLiNER2-detected entity type. The defaults come from tuning on the development
split of the NSS PII detection evaluation for exact-span F0.5, which weights
precision over recall. The checkpoint's scores cluster just below 1.0 for most
types, so most defaults are above 0.97. `ssn` is low because the SSN format
check, not the score, separates SSNs from other government IDs. Regex-owned
types have no confidence threshold. Checkpoint aliases use the threshold of the
NSS entity they normalize to.

Two further rules reduce false names, and the name thresholds were tuned with
both in place. First, NSS also asks GLiNER2 for the labels in
`name_exclusion_thresholds`, such as `medical_condition` and
`family_relationship`, and drops any detected name that overlaps one of those
spans. This removes words such as "mother" or "male" from the names found in
clinical text, but it can also drop a real name the model scores as, for
example, a `gender` term. Set `name_exclusion_thresholds: {}` to turn exclusion
off for text where missed names matter more than false ones. Second, with
`require_middle_name_neighbors`, a middle name is kept only inside a detected
full name or directly between a detected first and last name, which rejects
clothing sizes such as "medium" in product reviews.

NSS automatically uses CUDA when available and otherwise runs GLiNER2 on CPU.
It deduplicates identical cell text, flattens overlapping chunks across unique
values, and passes `batch_size` to GLiNER2's batched inference API. For CUDA
out-of-memory errors, reduce `batch_size` before changing chunk geometry. The
runtime logs model loading and inference progress without including source
text. See [Program Runtime -- GLiNER2 Device and Throughput](../user-guide/troubleshooting.md#gliner2-device-and-throughput)
for tuning and language-coverage guidance.

### Free-text replacement

Repeated accepted values reuse replacements within their configured record or
group scope. Free-text replacements follow the same [replacement
rules](#replacement-rules) as structured columns: each differs from the
detected text ignoring case, and is written in each occurrence's case.

A detected span whose text equals a value already known in the same record or
group always reuses that value's replacement. With `find_and_replace` (the
default), NSS also searches every free-text cell for those known values and
replaces the occurrences the detectors missed. Known values are
the original values of structured replacement columns, including the words of
a plain full name, and the values detected anywhere in that record's or
group's free text. For example, if the `first_name` column holds `Kenneth`,
then `kenneth called back` in the notes of that row is replaced even when the
detector misses it. The search matches whole words and ignores case, so
`Kenneth` matches `Kenneth's` but not `Kenneths`. It skips values shorter than
three characters, one-word values that are also common English words, such as
`Will`, `Brown`, or `June`, and missing-value placeholders such as `Unknown`,
because replacing those would also change ordinary words. A match is also
skipped when it overlaps a span GLiNER2 tagged with a `name_exclusion_thresholds`
label, so a structured last name `Parkinson` does not change
`Parkinson's disease` when the model tags it as a medical condition. When a
match overlaps a detected span, the longer span wins, and a detected span wins over a
match of equal length. Replacement traces mark matches with the detection
source `structured_value` or `detected_value`. Set `find_and_replace: false`
to replace only detected spans, for example to measure what the search adds. NSS does not replace undetected aliases, such
as nicknames.

Complete birth dates written in natural language, such as `5 April 1990` or
`April 5th, 1990`, are parsed strictly, shifted the same way as structured
dates, and rendered as ISO dates when no explicit strftime pattern exists.
Incomplete or vague candidates such as `spring`, and relative expressions such
as `yesterday`, `in 2 days`, or an age like `30 years`, are not treated as
dates. Such candidates are ignored and reported only as an aggregate warning
without the source text.

An independently detected name or street component may reuse a parent
replacement only when NSS can align it semantically. A full name, whether in a
structured column or detected in free text, aligns with first, middle, and
last names through explicit name-pattern placeholders, or, without a pattern,
when it is a two- or three-word name such as `Kenneth Gonzales`. A free-text
full name like that is replaced word by word, so `Kenneth Gonzales` and a bare
`Kenneth` in the same record or group get the same first-name replacement.
Titles and suffixes such as `Dr`, `Mrs`, `Prof`, `Jr`, or `III` are kept
unchanged and do not count as name words, so `Dr. Ada Lovelace` becomes
`Dr. <first> <last>`. A single-letter initial is replaced by a different
letter: when the record or group has exactly one replaced name of that kind
starting with the same letter, the initial takes the first letter of that
replacement, so after `Ada Lovelace` becomes `Grace Hopper`, `Ada L.` becomes
`Grace H.`. Initials are never reused on their own in other text. A name that
does not leave two or three words after removing titles and suffixes is
replaced whole. A street in free text reuses the street of a structured address
whose locality was kept, as described in [Full street
addresses](#full-street-addresses). NSS never guesses components from other
token positions or isolated house numbers.

### Full street addresses

A `street_address` column may hold a full address such as
`489 Park Blvd, Mesa, IA 76237`. NSS runs the GLiNER2 detector on each distinct
value, and when the only detected entity is a street at the start of the value,
followed by a comma or space, it replaces only that street and keeps the rest as
written: `489 Park Blvd, Mesa, IA 76237` becomes, for example,
`67911 Johnson Knoll, Mesa, IA 76237`. This works for any address format the
detector recognizes, such as `Bahnhofstraße 21, 10115 Berlin, Germany`; the new
street comes from the configured sampler and locale. Every other value,
including one that starts with a unit outside the detected street or one the
detector does not recognize, is replaced whole with a generated street. If the
detector cannot run, for example because the `gliner2` package or model is
unavailable, NSS logs a warning and replaces every street-address value whole.

Keeping the locality means the original city, state, postal code, and country
remain in the output, as they already do in separate city, state, or postal-code
columns that are not replaced and in free text around a detected street.
Combined with other columns, such as a birth date, a postal code can help
re-identify a person.

### Standalone replacement

Apply replacement to a complete input dataframe without running holdout,
training, generation, or evaluation:

```bash
safe-synthesizer run replace-pii \
  --config config.yaml \
  --data-source data.csv \
  --output-file pii_replaced.csv
```

When `--output-file` is omitted, the command adds `_pii_replaced` to the input
stem and writes the result beside `safe-synthesizer-config.yaml` and
`pii_replacement_plan.yaml` in the run directory. For example, `data.csv` produces `data_pii_replaced.csv`, and
`data.parquet` produces `data_pii_replaced.parquet`. Output is written as
Parquet when its path ends in `.parquet`, which keeps every column's stored
type, and as CSV otherwise. The input can be CSV, TXT, Parquet, JSON, or JSONL,
from the CLI or from a path passed to the SDK's `with_data_source`. The
resolved configuration and replacement file are always written, with the review
comments described under [Plan-only workflow](#plan-only-workflow), so the exact
plan and sampler mappings can be reviewed and reused. Replacement maps and raw detector results
are not persisted.

For CSV and TXT inputs, every cell that replacement leaves unchanged is written
exactly as it appears in the input file. For example, ZIP code `02054` stays
`02054` rather than becoming `2054`, and `N/A` stays `N/A` rather than
becoming an empty cell. The file is read once, and the unchanged cells come
from those same bytes, so editing the file during a long run cannot mix rows
from two versions. Local paths and HTTP(S) URLs are read this way; for other
URLs, such as `s3://`, the text is read again when the output is written. The
SDK does the same when `with_data_source` is given a CSV path or URL; for a
dataframe source, pass the input read as text with
`source_text=pd.read_csv(path, dtype=str, na_filter=False)`.

The SDK equivalent returns the complete `TransformResult` and writes files only
for paths supplied by the caller:

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

## Self-hosted LLM

Automatic discovery can use an LLM to classify columns, propose patterns,
choose dependencies, and map dependency values; see
[How discovery uses the LLM](#how-discovery-uses-the-llm). This part covers the
`llm` configuration, where planning requests go, and the local server NSS can
run for them.

### LLM configuration

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
endpoint, and stops it before planning returns, including when planning fails. Plan validation, replacement, and
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

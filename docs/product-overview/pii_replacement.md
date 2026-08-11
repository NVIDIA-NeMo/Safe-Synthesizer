<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

PII (Personally Identifiable Information) replacement detects and replaces sensitive
information in your datasets before synthesis. It reduces the model's exposure to
**detected** PII (e.g. names, addresses, identifiers); it is not a guarantee that every
sensitive value is found or correctly typed. Heuristic discovery can miss columns or
assign an unexpected entity type, so always review the emitted
`pii_replacement_plan.yaml` before training.

## How It Works

PII replacement runs in three stages:

1. Discovery: Inspects the dataframe and builds a *replacement plan* -- which columns
   hold PII, which entity type each one holds, and which columns describe the same
   person. Supply your own plan to skip this.
2. Validation: Checks the plan against the dataframe, so unknown columns or duplicate
   entries fail before any data is touched.
3. Replacement: Draws a synthetic identity per persona and substitutes values, then
   propagates substitutions from structured entity columns (persona-backed or
   standalone) into free-text columns so prose agrees with those columns. Free-text
   scanning is skipped in heuristic mode only when no structured entity columns
   were identified.

## Detection

Discovery reads column names, values, and dtypes -- it does not call an LLM or an NER
model. A **replaceable** entity is assigned only when the column **name** matches a known
pattern (regex or fuzzy); value evidence alone never plans replacement. Where a simple
content check exists (email/phone/SSN/card/IP regexes, credential-like API keys, street
house numbers, parseable birth dates), that check is also required. Hand-written plans
remain the escape hatch for oddly named columns. Generic temporal columns
(`date` / `datetime` / `time` / `duration`) are still identified from values alone so
they can be kept out of free-text scanning without being replaced.

| Signal | What it examines |
|--------|------------------|
| Column name | Required for replaceable entities. Fuzzy match against known name patterns (`dob`, `surname`, `postcode`, `ssn`, ...) |
| Value pattern | Content gate when a simple check exists. The dominant concrete format must cover at least 85% of non-null values. Without a supporting name label, value hits are ignored for replacement (temporals are the exception) |
| Group constancy | With `data.group_training_examples_by` set, columns that stay constant within a group are treated as attributes of that group's entity; uniqueness for those columns is measured per group |
| Field type | Long free-form columns are marked `free_text` and scanned for known values instead of being replaced as a single value |

!!! note "LLM-assisted discovery"
    `replace_pii.llm_enhancement: true` is reserved for a future release and currently
    raises `ParameterError` from the `PiiEnhancer` seams (`enrich_structured`,
    `enrich_demographics`, `detect_freetext_spans`). Discovery is heuristic in
    this release; when LLM mode lands, heuristics will pass candidates and decision
    context into those seams and the LLM will be the final judge -- without forking
    apply. Demographics: the LLM may infer sex and fine-grained
    `ethnic_background` from names, starting from any heuristic
    `match_persona_by` values.

## Replacement Methods

| Method | Applies to | Behavior |
|--------|-----------|----------|
| Synthetic persona | Persona-sourced columns | One synthetic identity supplies persona-sourced columns under a persona, so first name, last name, and email belong to the same fictional person, each written the way its own column writes it |
| Pattern-preserving substitution | Entity-driven columns | The synthetic value is derived from the original, keeping its format (a `%m/%d/%Y` date stays `%m/%d/%Y`, a `+1-###-555-####` phone stays `+1-###-555-####`), and is stable per distinct original value within the configured scope |
| Free-text propagation | `free_text` columns | Values replaced from structured entity columns (persona-backed or standalone) -- plus individual name tokens, so honorifics and partial mentions follow -- are swapped inside the text. In heuristic mode, free text is scanned only when at least one structured entity column was identified (otherwise there is nothing to propagate). Matches are case-insensitive; the synthetic is reshaped to the matched token's case. A token is read as free text mentions it, without the punctuation its column writes around it, so a note naming `SMITH` alone is rewritten whether the column writes `Jane Smith` or `SMITH, Jane` |
| Identified, not replaced | Generic `date` / datetime / time / duration, and address parts (`city`, `state`, `zipcode`) | Reported in the plan logs and deliberately left untouched |

### Consistency scope

`scope` controls how widely one original value keeps the same synthetic value:

| Scope | Meaning |
|-------|---------|
| `record` | Independently per row. Builds one replacement map per row for standalone identifiers, which is fine for typical training samples but can be costly on large frames (a user warning fires above 25k rows) — prefer `group` or `dataframe` when per-row independence is not required |
| `group` | Consistent within each training group (requires `data.group_training_examples_by`), so a patient's rows agree with each other |
| `dataframe` | One mapping across the whole dataset |

## Supported Entity Types

`entity_type` accepts the following closed vocabulary. Persona-sourced entities come
from a synthetic identity; entity-driven entities are derived from the original value
and never depend on a persona, so listing one under a persona changes nothing.

| Entity type | Kind | Notes |
|-------------|------|-------|
| `first_name`, `middle_name`, `last_name`, `full_name` | Persona-sourced | Conditioned on `match_persona_by` attributes when present, and written the way the column writes names (`SMITH, Jane`). No persona source carries a middle name, so one is drawn per person, of that person's sex, and is the same wherever it appears |
| `email` | Persona-sourced | Built from the same synthetic identity as the names, following the convention the value itself follows (`j.smith@acme.com`), and keeping its domain. A value that reads as no person is a handle, which is regenerated in its own shape instead |
| `phone_number` | Entity-driven | Rebuilt in the column's own format. Persona-sourced only under the internal `pgm` backend, the one persona source that carries a number, whose number is then printed in that same format |
| `street_address` | Persona-sourced | |
| `city`, `state`, `zipcode` | Special | Name-matched address parts: identified and **not** replaced (not allocated as persona fields) |
| `ssn`, `national_id` | Entity-driven | Rebuilt from the original (Faker / shape-preserving); not carried on managed or PGM personas |
| `date_of_birth` | Entity-driven | Perturbed per record or group, keeping the original date format |
| `unique_identifier` | Entity-driven | Record and group IDs |
| `credit_debit_card` | Entity-driven | |
| `api_key` | Entity-driven | |
| `ipv4`, `ipv6` | Entity-driven | |
| `date` | Special | Marks a column as a generic (non-birth) date: structured, identified, and **not** replaced |
| `free_text` | Special | Marks a column for value propagation rather than whole-column replacement |

To change how a column is treated -- including forcing a column to be skipped by
leaving it out -- edit the replacement plan rather than the entity vocabulary, which is
fixed in this release.

## Configuration

PII replacement is on by default. Set `replace_pii: null` in YAML (or pass
`--no-replace-pii` on the CLI, or `.with_replace_pii(enable=False)` in the SDK) to turn
it off. Every field below has a usable default, so the block is only needed when
customizing. For the full schema, refer to
[Configuration Reference -- Replacing PII](../user-guide/configuration.md#replacing-pii).

```yaml title="replace_pii section"
replace_pii:
  # "auto_discovery" (default), a path to a plan file, or an inline plan.
  replacement_plan: auto_discovery
  replacement:
    locale: en_US
    seed: 42
  person:
    # managed (default) or faker
    backend: managed
```

Time-series group keys, `data.order_training_examples_by`, and the time-series
timestamp column are never replaced (they define training structure). Auto-discovery
omits them from the plan; a user-supplied plan that lists them under
`columns_to_replace` is rejected.

### Persona backends

`person.backend` chooses where synthetic identities come from. `managed` (the default)
samples them from Nemotron-Personas locale parquet files under the managed-assets root
(download with the NGC CLI -- see
[Running -- Managed persona assets](../user-guide/running.md#managed-persona-assets)).
`faker` generates them with Faker and needs no assets. A third
value, `pgm`, drives an internal probabilistic generator that is not distributed with
Safe Synthesizer; it needs a local source tree and fails the run with an error if that
tree is missing, rather than quietly substituting another backend.

`match_persona_by` may list `sex` and, under `managed` / `pgm`, `ethnic_background`.
Faker only conditions given names on sex, so auto-discovery omits `ethnic_background`
when `person.backend` is `faker`, and a hand-written matcher for it is ignored (with a
plan advisory). The same applies when managed assets are missing and the engine falls
back to Faker for that run.

The backend also decides where phone numbers come from. Only the `pgm` generator
produces one (with an area code drawn from the persona's own address), so under the
other backends a phone column is replaced on its own, from the column's format, rather
than from the persona. This is the one entity whose kind depends on configuration, and
it applies to plans you write as well as discovered ones: a phone column listed under a
persona is still replaced standalone unless the backend is `pgm`. Either way the column
keeps the format it had, so the choice of backend does not change how the data reads.

### Replacement plan

By default the plan is discovered automatically and written to
`pii_replacement_plan.yaml` in the run directory. Pass an edited copy back through
`replace_pii.replacement_plan` (as a path or inline mapping) to take control.

A plan has two sections. `persona_backed_columns` holds the columns that describe a
person: each entry is one persona whose `columns_to_replace` are filled from a single
synthetic identity, so the values stay consistent with each other. Auto-discovery names
personas from column-name role prefixes when possible (`patient_name` → `patient`,
`provider_email` → `provider`); unprefixed columns use `person_1`, `person_2`, …. On a
role collision (duplicate entity or disagreeing name parts), discovery emits
`patient_2` / `provider_2` and warns.
`standalone_columns_to_replace` holds columns replaced on their own, with no persona
behind them (record IDs, free-text notes, cards, IPs, and similar).

The engine follows **entity type**, not YAML section: a free-text or entity-driven
column under `persona_backed_columns` is still replaced standalone, and a
person-identifying column under `standalone_columns_to_replace` does not share a
synthetic person with other columns. Preflight and apply log warnings for those
mismatches; they do not rewrite the plan.

```yaml title="pii_replacement_plan.yaml"
# How widely one original value keeps the same synthetic value:
# record (per row), group (per training group), or dataframe (whole dataset).
scope: group

persona_backed_columns:
  - persona: patient
    columns_to_replace:
      - column_name: first_name
        entity_type: first_name
      - column_name: last_name
        entity_type: last_name
    # Existing columns that constrain which persona is drawn. Read, never replaced.
    match_persona_by:
      - persona_attribute: sex
        column_name: sex
  - persona: provider
    columns_to_replace:
      - column_name: provider_name
        entity_type: full_name

standalone_columns_to_replace:
  - column_name: patient_id
    entity_type: unique_identifier
  - column_name: notes
    entity_type: free_text
```

A column may also carry `patterns`: formats that preserve how values are written
(most common first). A value is rewritten with the first listed pattern that
describes it; a value no pattern describes keeps its own shape.

What a pattern means depends on the entity:

- Persona-sourced names and email -- placeholders for person parts (`{First}`,
  `{last}`, `{f}.{last}@{domain}`, and so on), assembled from the synthetic
  persona rather than character templates. Email detection is ASCII-only in this
  release (IDN / Unicode local parts are not matched).
- `date_of_birth` -- strftime formats such as `%m/%d/%Y`.
- Identifiers, cards, and phones -- value templates (`#` digit, `^` A-Z, `@`
  a-z, `*` alphanumeric, `[68]` choices). Under `pgm`, phone digits can come
  from the persona while the template only controls punctuation.
- No patterns -- `ipv4` / `ipv6`, `ssn`, `national_id`, `street_address`, and
  `free_text` (and listing a pattern on them is an error).

Discovery infers patterns from the column; edge cases (evidence thresholds, tails,
Luhn checksums for cards) live in plan YAML comments and validation errors rather
than here. Plans are validated against the dataset before any replacement runs, so
columns that are not in the data, missing `entity_type`, patterns the column's own
values do not match, and patterns set on an entity that does not use them, are
reported as errors. A plan you supply is checked during pre-flight, which means
`--validate` catches these without running the pipeline and reports every problem
in the plan at once.

## Auto-discovery expectations (MVP)

Always review the emitted `pii_replacement_plan.yaml` before production or regulated
use. Auto-discovery works best on English, US-ish, well-named, high-cardinality
tabular schemas. Expect to hand-author plans for narrative-heavy data, international
IDs/phones, low-N pilot slices, and opaque tokens. Contiguous sequential integer
keys (any origin, e.g. ``1,2,3,…`` or ``100000,100001,…``) are skipped; gapped
numeric IDs can still be planned. Free text propagates values replaced from
structured entity columns (persona-backed or standalone) in heuristics mode;
when no structured columns are found, free text is not scanned — it is not a
redaction NER. Generic (non-birth) date columns are identified
and left unchanged so temporal relationships stay intact. Street-name-only columns
(no house number) are not planned as `street_address`.

## When to Use PII Replacement

Consider using PII replacement when:

- Your data contains names, addresses, or other direct identifiers
- Compliance requires PII removal before processing
- You want to ensure the model cannot memorize sensitive values
- You need to share synthetic data with external parties

PII replacement is on by default as a pre-processing step before synthesis.

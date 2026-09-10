<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII v3 Replacement Engine Handoff

## Purpose and next-session starting point

This document hands off the agreed design for implementing PII v3 replacement execution. It is intended to be
self-contained enough for a new session to continue without relying on the conversation that produced it.

Interface definitions are tracked in [PR 741](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/pull/741). Start the
next implementation branch from `yunfeng/pii-v3-replacement-interfaces` while that PR is open, or from the branch that
contains it after merge. The next session implements replacement execution behind those interfaces; it should not
redesign them without a concrete implementation blocker.

The interface work was originally based on:

```text
yunfeng/pii-v3-llm-enhancement
dc954ef446fbd0d8ce197f1e877fa4271db0d27e
```

That target implements the v3 configuration, flat plan DAG, validation, and LLM-assisted plan-discovery
structure, but it does not implement replacement execution. Its heuristic plan discoverer remains a no-op. This
handoff covers replacement execution only; heuristic plan discovery is separate work.

The first implementation uses GLiNER2 plus deterministic regex for free-text entity detection. LLM-based free-text
entity detection is deferred. The local LLM on the target branch still performs automatic column classification and
plan discovery when configured.

## Reference map

### V3 behavior notes

Primary behavior notes:

<https://github.com/user-attachments/files/30950534/pii-repl-v3-behavior-impl-notes.md>

The notes define the intended v3 behavior but omit the LLM plan-discovery path. The target branch supplies that path.
Use the notes and this handoff as the behavior specification, with this handoff recording decisions made after the
notes were written.

### Target branch

Reference: `yunfeng/pii-v3-llm-enhancement` at `dc954ef446fbd0d8ce197f1e877fa4271db0d27e`.

Important files on that ref:

- `src/nemo_safe_synthesizer/config/replace_pii.py`: v3 entity catalog, flat plan DAG, scopes, dependency validation,
  sampler configuration, and the current `LLMConfig`.
- `src/nemo_safe_synthesizer/pii_replacer/llm_client.py`: OpenAI-compatible transport, inference-setting precedence,
  and strict JSON-schema response support.
- `src/nemo_safe_synthesizer/pii_replacer/planning/`: plan assembly, LLM enhancement, validation, persistence, and the
  no-op heuristic discoverer.
- `src/nemo_safe_synthesizer/pii_replacer/transform_result.py`: the existing `TransformResult` and
  `ColumnStatistics` models.
- `tests/pii_replacer/planning/` and `tests/sdk/test_pii_planning.py`: established planning tests and seams.

PR 741 makes `ReplacePiiConfig.llm` planning-only. Free-text replacement does not initialize or use that LLM, and an
explicit replacement plan does not require one.

Do not replace the flat v3 configuration with historical persona or standalone sections. Relationships between planned
columns are expressed by `depends_on`, and execution follows that DAG.

### Main branch detector

Reference: `main` at `6d1849197d9580df9da4c763235f9e180c4e13b9`.

Main contains the prior production GLiNER pipeline:

- `src/nemo_safe_synthesizer/pii_replacer/data_editor/detect.py`: GLiNER loading, batching, chunking, caching, label
  conversion, and span production.
- `src/nemo_safe_synthesizer/pii_replacer/ner/regexes/`: historical regex predictors.
- `src/nemo_safe_synthesizer/config/replace_pii.py`: historical detector defaults.

Treat this implementation as reference evidence only. The v3 detector is a new adapter around GLiNER2 and the regex
work from Anonymizer PR 265. Do not retain the old v2 configuration or data-editor pipeline merely for compatibility.

### NVIDIA-NeMo/Anonymizer

Regex implementation:

<https://github.com/NVIDIA-NeMo/Anonymizer/pull/265>

Reference the PR at head commit `05c1a9ccfd68c4aaf56e4a84c0b3d9acc07bacac` until it merges, then update the
reference to the merge commit. Relevant behavior includes:

- built-in regex rules and deterministic local validators;
- match timeouts and per-record match limits;
- stable rule identifiers and source provenance;
- normalized character-span production;
- deterministic source merging and overlap resolution.

The PR currently includes built-ins for credit/debit cards, email, IPv4, IPv6, MAC addresses, and URLs. The v3 entity
catalog directly supports credit/debit cards, email, IPv4, and IPv6. Do not attribute phone, SSN, or API-key patterns
to this PR. MAC address and URL require a separate entity-catalog decision.

Do not adopt the PR's optional LLM validation path in the initial implementation.

Overlap resolver:

<https://github.com/NVIDIA-NeMo/Anonymizer/blob/05c1a9ccfd68c4aaf56e4a84c0b3d9acc07bacac/src/anonymizer/engine/detection/postprocess.py#L240-L297>

Anonymizer also expands some non-regex entity values to matching occurrences with string search:

<https://github.com/NVIDIA-NeMo/Anonymizer/blob/05c1a9ccfd68c4aaf56e4a84c0b3d9acc07bacac/src/anonymizer/engine/detection/postprocess.py#L347-L382>

Do not adopt occurrence expansion. Safe Synthesizer will replace only accepted detector spans.

### GLiNER2 model

Use
[fastino/gliner2-privacy-filter-PII-multi](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
as the default checkpoint. Keep the model ID configurable.

The base [fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1) checkpoint remains useful as a
comparison in opt-in live-model evaluations, but it is not the default.

### Nina's previous v3 implementation

Reference: `nina-xu/pii-plan-preview` at `d4cd1b3d04cea64ede626d02c97692c82bfe712c`.

Useful reference areas:

- `src/nemo_safe_synthesizer/pii_replacer/replacer.py`: orchestration and statistics assembly.
- `src/nemo_safe_synthesizer/pii_replacer/replacement/`: structured replacement, scopes, managed/Faker behavior,
  component mappings, and free-text editing.
- `tests/pii_replacer/replacement/`: behavior examples for scoping, free text, demographics, phone numbers, and
  consistency.

This is reference evidence, not a parity goal. Its persona/standalone plan shape is obsolete. It has no working
replacement-time LLM detector to transplant.

### Internal LLM detection notebook

Notebook:

<https://gitlab-master.nvidia.com/sdg-research/na-for-nss/-/blob/main/dd_person_sampling_pii_replacement_v2.ipynb>

Implementation imported by the notebook:

<https://gitlab-master.nvidia.com/sdg-research/na-for-nss/-/blob/main/pii_replacement.py#L2203>

The notebook asks an LLM for verbatim entity values, locates them in text, and makes a second LLM call to author
replacements. Neither behavior is part of the initial implementation. Retain the notebook only as evidence for a future
LLM span-detection experiment.

## Settled product decisions

- Automatic LLM-assisted column classification and plan discovery run against the full input dataset.
- Replacement runs after holdout creation and transforms only the training split.
- Initial free-text detection always combines GLiNER2 with applicable deterministic regex rules.
- The configurable default GLiNER2 checkpoint is `fastino/gliner2-privacy-filter-PII-multi`.
- There is no LLM free-text detection method in the initial implementation.
- There is no propagation-only mode. Every planned free-text target uses the GLiNER2-plus-regex detector pipeline.
- Only accepted GLiNER or regex spans are eligible for free-text replacement.
- Never search for other occurrences of a detected value, a structured value, or an alias.
- Component mappings may select the replacement for an independently detected span; they never create spans.
- Follow Anonymizer's deterministic longer-span-first overlap policy.
- Do not adopt Anonymizer's string-based occurrence expansion.
- Managed sampling and Faker remain the replacement backends. LLM sampling is out of scope.
- `ManagedReplacementGenerator` and `FakerReplacementGenerator` are the two adapters at the replacement-generator
  seam.
- The flat `depends_on` DAG is the execution architecture.
- Record mappings use stable positional row identity. Dependencies are generation inputs, not record mapping identity.
- For group-scoped mappings, group consistency wins over dependency consistency.
- The first group occurrence establishes the replacement. If later dependency values differ, reuse the first
  replacement and emit an aggregated, PII-free warning.
- Nina's previous v3 implementation and main's detector are references, not compatibility targets.
- The complete original-to-replacement map is not persisted by default; explicit secure persistence remains an open
  product decision.
- Dataframe scope is not supported initially. If added later, it follows group-style first-occurrence-wins behavior.
- Repeated accepted detections of the same entity and exact value reuse one replacement across all planned free-text
  columns within the row or group scope, independently of propagation mappings.
- Structured mapping values use a type-tagged canonical representation. Canonicalization is stable typed
  serialization, not trimming, case-folding, or other text cleanup.
- The first PR defines interfaces only so reviewers can evaluate the overarching design before implementation.

## Interface PR contract

PR 741 settles configuration, types, invariants, and the external seam. It intentionally does not perform replacement.

### Public replacement interface

Define:

```python
class TabularPiiReplacer:
    def __init__(
        self,
        config: ReplacePiiConfig,
        *,
        data_config: DataParameters,
        time_series: TimeSeriesParameters | None = None,
    ) -> None: ...

    def replace(self, df: pd.DataFrame) -> TransformResult: ...
```

The intended contract is:

- `replace()` returns a new dataframe and does not mutate the caller's frame.
- The module owns DAG execution, scope keys, generation, detection, overlap handling, and statistics.
- `replace()` does not write artifacts.
- The pipeline decides whether and where to persist the resolved plan.

Extend `TransformResult` with:

```python
replacement_plan: PiiReplacementPlan
generation_statistics: ReplacementGenerationStatistics
elapsed_time_seconds: float
```

`ReplacementGenerationStatistics` records elapsed generation time and the number of distinct replacements generated
after cache reuse.

Do not add a persisted replacement-map path to this interface. If an in-memory mapping is needed, design it as an
explicit sensitive result with a separate persistence decision.

### Replacement generator seam

`ReplacementGenerator.generate(ReplacementGenerationRequest) -> str` is the internal generation interface. Equal
requests must produce equal results. The request carries entity type, generator-facing original value, effective
dependencies, optional pattern, and derived seed. Mapping scope, cache reuse, timing, and dataframe mutation remain in
the replacement executor.

`ManagedReplacementGenerator` and `FakerReplacementGenerator` are the two adapters. Each declares its
`PiiSamplerBackend` and accepts `PiiReplacementSettings` plus `PiiSamplerConfig` at construction. Implement their
generation behavior in the follow-up without widening the public `TabularPiiReplacer` constructor.

### Free-text configuration

Replace the old proposal's `llm`/`fast` method union with one initial GLiNER2-plus-regex configuration:

```python
class FreeTextDetectionConfig(NSSBaseModel):
    model_id: str = DEFAULT_GLINER2_MODEL_ID
    threshold: float = 0.3
    batch_size: int = 8
    chunk_length: int = 384
    chunk_overlap: int = 128


class ReplacePiiConfig(Parameters):
    free_text_detection: FreeTextDetectionConfig = Field(default_factory=FreeTextDetectionConfig)
```

PR 741 provisionally uses the base checkpoint. Before implementing model loading, the follow-up must change
`DEFAULT_GLINER2_MODEL_ID` and the documented configuration example to
`fastino/gliner2-privacy-filter-PII-multi`. Validate `threshold` within `[0, 1]`, positive batch/chunk values, and a
nonnegative overlap smaller than the chunk length.

Built-in regex runs as part of this detector pipeline. Do not expose an LLM method. Whether reviewers want an explicit
regex disable switch is an open first-PR question; the minimal interface has no switch.

`ReplacePiiConfig.llm` remains the existing opt-in for LLM-assisted plan discovery. Revise its descriptions so it no
longer claims to configure replacement-time detection. An explicit replacement plan does not require initializing the
LLM.

### Internal detection contracts

Define immutable types sufficient to separate model inference from replacement resolution:

```python
@dataclass(frozen=True, slots=True)
class DetectionCellId:
    row_position: int
    column_name: str


@dataclass(frozen=True, slots=True)
class DetectionCell:
    cell_id: DetectionCellId
    text: str
    allowed_entity_types: frozenset[EntityType]


@dataclass(frozen=True, slots=True)
class DetectedSpan:
    cell_id: DetectionCellId
    start: int
    end: int
    entity_type: EntityType
    source: Literal["gliner", "regex"]
    score: float | None = None
```

The exact class names may change during review, but the contracts must preserve these invariants:

- Cell identity is positional and remains safe with duplicate dataframe indexes.
- Cell identity contains no raw cell value.
- Offsets are half-open and relative to the original complete cell.
- `0 <= start < end <= len(cell.text)`.
- `cell.text[start:end]` is the exact detected occurrence.
- Each span carries normalized v3 entity type and provenance.
- Detector results cannot contain replacement values.

Keep the detector and overlap-resolver seams private to the replacement module. The initial detector is composite:
GLiNER2 and regex feed the shared span resolver.

### Mapping, canonicalization, and warning contracts

Structured record values use:

```text
target column
+ stable row position
+ canonical original value
```

Dependencies affect generation but are not part of record mapping identity. A row has one effective dependency tuple
for a given target.

`CanonicalValue` is a type-tagged, normalized identity for a structured scalar. The type tag prevents unlike values
such as integer `1` and string `"1"` from sharing a mapping. Its payload is a deterministic, locale-independent string.
Canonicalization preserves string content exactly: it does not trim, case-fold, or otherwise clean text. Equivalent
Python, NumPy, and pandas scalars must produce the same type-tag/payload pair. Missing values are preserved and never
canonicalized.

Detected free-text values use exact strings rather than `CanonicalValue`:

```text
+ scope identity
+ detected entity type
+ exact original substring
```

The free-text key intentionally omits the target column. Equal accepted detections therefore reuse one replacement
across all planned free-text columns within a row or group. Every occurrence must still have its own accepted detector
span; mapping reuse never creates spans.

Group scope:

```text
target column
+ original group identity
+ canonical original value
```

For group scope, store the effective dependency tuple used by the first occurrence as mapping provenance, not identity.
Define a structured warning/statistic for later dependency drift. It should expose only safe aggregate information such
as target column, conditioner entity types, and conflict count. It must not expose original values, group identifiers,
or dependency values.

Stable positional row order defines which occurrence is first.

Dataframe scope is not part of the initial implementation. A future dataframe scope should use the same
first-occurrence-wins policy as group scope rather than adding dependencies to mapping identity.

### Interface PR scope

Include configuration validation, immutable contract types, public docstrings, and focused tests for those contracts.
Do not yet:

- add or load GLiNER dependencies;
- copy regex rules from Anonymizer;
- implement inference, overlap resolution, entity generation, or text construction;
- wire replacement into the pipeline;
- persist new artifacts.

## Replacement engine behavior for follow-up PRs

### DAG execution and consistency

- Resolve and validate the plan, compile a stable topological order, and execute targets against a dataframe copy.
- Read dependencies from the working frame after upstream replacement.
- Maintain an internal positional row identity so duplicate input indexes remain safe and unchanged.
- Snapshot original group keys before replacement so replacing the group column cannot change scope membership.
- Preserve missing values, row order, column order, protected values, and unplanned columns. A replaced column may
  become string/object dtype.

For record scope, the mapping key uses target column, stable row position, and canonical original value. Dependencies
remain generation inputs. For group scope, the first replacement wins:

1. Encounter the first `(target column, original group, canonical original)` mapping in positional row order.
2. Generate its replacement using that row's effective dependencies.
3. Store both the replacement and the initial dependency tuple.
4. Reuse the replacement for later occurrences of that mapping.
5. When a later dependency tuple differs, increment an aggregate conflict count and warn; do not regenerate or fail.

Plan-schema errors remain `ParameterError` conditions. Dependency drift among rows in a valid group is runtime data
quality information, not invalid configuration.

### Deterministic generation

- Resolve the base seed from explicit replacement configuration, then `PERSON_RANDOM_SEED`, then `42`.
- Derive a per-mapping RNG seed using SHA-256 over the base seed, complete mapping key, and operation-purpose tag.
- Use randomness for managed/Faker choices, masks, birth-date shifts, retries, and collision avoidance.
- Given the same plan, accepted detector spans, and seed, output must be deterministic.
- Support managed and Faker person sampling, name patterns, character masks, plus-or-minus 365-day birth-date shifts,
  Luhn-valid cards, IP addresses, and collision-resistant identifiers.
- If managed sampling fails, warn once per affected category and fall back deterministically to Faker.
- Require generated values to differ from originals and satisfy entity-specific constraints.
- Use one pattern parser for validation and rendering so their grammars cannot drift.

### Email behavior

- `{domain}` means the original email domain.
- `{organization}` uses the effective value of an organization dependency.
- Literal suffixes remain literal, including `{organization}.com`, `{organization}.org`, and
  `mail.{organization}.co.uk`.
- Normalize organizations into lowercase DNS labels: transliterate when possible, replace non-alphanumeric runs with
  hyphens, retain all words, do not guess legal-suffix removal, and enforce the 63-character DNS-label limit.
- If `{domain}` or `{organization}` cannot be resolved, generate a deterministic compatible value and issue an
  aggregate PII-free warning.
- Without an email pattern, generate the complete synthetic email address and domain through the sampler.

## Free-text detection and replacement

### GLiNER2 adapter

- Load GLiNER2 lazily only when the resolved plan contains a free-text target.
- Run inference over unique original texts using configured batching and overlapping chunks.
- Convert chunk-relative offsets to original-cell offsets before emitting `DetectedSpan`.
- Normalize model labels into the v3 `EntityType` catalog.
- Filter results to entity types allowed for fresh replacement by the plan.
- Exclude `free_text`, identify-only types, and `unique_identifier` unless product policy changes.
- Raise `GenerationError` on model-load or inference failure; do not silently continue as regex-only.
- Add model dependencies through `cuda_deps.toml` and regenerate the dependency blocks and lockfile through the
  repository's documented workflow.

Use the selected PII-specialized default for implementation. Compare the base checkpoint on repeated strings,
ambiguous substrings, nested entities, Unicode, long cells, and chunk boundaries only in opt-in live-model
evaluations outside deterministic CI.

### Regex adapter

Adapt the deterministic portions of Anonymizer PR 265:

- built-in rule registry;
- local structural validators, including Luhn and IP validation;
- bounded matching with timeouts and per-rule/per-record limits;
- stable rule identifiers and provenance;
- exact regex match offsets.

Initial applicable v3 labels are `credit_debit_card`, `email`, `ipv4`, and `ipv6`. Do not add MAC or URL
silently. Do not claim phone, SSN, or API-key coverage from this PR. Do not enable its optional LLM validation.

### Overlap resolution

Follow Anonymizer's deterministic policy:

1. Coalesce exact duplicate `(entity_type, start, end)` spans.
2. Rank candidates by descending span length, earlier start, and earlier end.
3. For otherwise tied GLiNER candidates, prefer higher confidence.
4. Apply source priority and stable entity-catalog order only for remaining ties. Give structurally validated regex
   priority over GLiNER on an exact tie.
5. Greedily accept candidates that do not overlap an already accepted span.
6. Return accepted spans sorted by `(start, end, entity_type)`.

Use half-open overlap:

```python
left.start < right.end and right.start < left.end
```

Touching spans do not overlap. Longer spans win genuine overlaps even when the shorter span came from regex.

Chunk-overlap duplicates must be converted to cell-relative offsets before this resolver runs. If a GLiNER adapter
performs an earlier same-source deduplication pass, its behavior must preserve these global ordering rules.

### Span-only replacement

Only accepted GLiNER or regex spans create free-text work:

- Do not locate other occurrences of `cell.text[start:end]`.
- Do not scan free text for structured-column values.
- Do not scan for name aliases or address components.
- Do not replace an undetected substring merely because its value is known.
- Do not adopt Anonymizer's `expand_entity_occurrences` behavior.

Each repeated occurrence must be independently detected and accepted. Once accepted, occurrences with the same entity
type and exact value reuse one replacement across all planned free-text columns within the row or group scope.

Structured and component mappings still support replacement consistency after detection. A separately detected name or
address component may reuse its parent's synthetic component when it matches a scoped mapping. The mapping does not
make an undetected occurrence eligible.

Example: if GLiNER detects both `John Smith` and a separate `Smith`, the name handler can reuse the same synthetic
surname. If the separate `Smith` was not detected, it remains unchanged. Apply the same rule to address components
that can be aligned reliably. Do not create ambiguous component mappings such as an isolated house number.

Names and validated address components may be composite. Email domains may be exposed only when the selected email
behavior changes the domain. Phone numbers, dates, cards, identifiers, IP addresses, and API keys remain atomic by
default. Never infer components by arbitrary token position.

All offsets refer to original cell text. After overlap resolution, build the result once in ascending span order by
appending unchanged gaps and generated replacements. Do not repeatedly mutate the input string or recalculate offsets.

Fresh spans use the same programmatic generators and configured scope as structured replacements. A shared free-text
target identity keeps the same detected original and entity type consistent across planned free-text columns within
that scope, subject to group first-replacement-wins behavior.

## Future LLM entity detection

LLM entity detection is a future experiment, not an implementation requirement. It is separate from the target
branch's existing LLM column classification and plan discovery.

Before adding an LLM detector, demonstrate that a small local model reliably returns:

- opaque cell identity;
- half-open `start` and `end` offsets into the submitted original text;
- a closed-vocabulary v3 `entity_type`.

Evaluate exact offset validity, precision, recall, and latency on repeated identical strings, substring collisions,
Unicode, nested and overlapping entities, long cells, and chunk boundaries. Validate offsets and labels locally. Do not
fall back to verbatim-value string search when offsets are missing or invalid.

If an LLM adapter is added later, derive its request budget from the selected model's context and token budget rather
than a fixed byte limit. The earlier 48 KiB proposal is removed.

## Pipeline integration and reporting

- Automatic LLM plan discovery reads the full input dataset.
- Replacement runs after holdout and transforms only the training split.
- Retain original training and test frames for evaluation.
- Persist the resolved plan as `<run_dir>/pii_replacement_plan.yaml`.
- Populate `ColumnStatistics` for every planned target:
  - Structured counts include every non-missing occurrence.
  - Detected-value sets contain unique originals.
  - Free-text counts reflect accepted detector spans that were actually replaced.
  - Planned targets with no matches still receive assigned type/entity metadata and empty detection maps.
  - Group dependency-drift counts are aggregated without raw values or group identifiers.
- Populate `ReplacementGenerationStatistics` with generation-phase elapsed time and the number of distinct generated
  replacements after cache reuse.
- Never log raw PII, detector inputs, detector outputs, conflicting values, or replacement maps.
- Raise `ParameterError` for invalid configuration, plans, dependencies, and patterns.
- Raise `GenerationError` for detector failures or unrecoverable generation failures.
- Verify identical shape and index, unchanged protected values, no unplanned edits, and statistics for every planned
  target before returning.

Do not persist the complete original-to-replacement map by default. If product requirements later require persistence,
make it explicit opt-in, label the artifact as sensitive, and define access controls and retention behavior.

## Implementation sequence after the interface PR

1. Change `DEFAULT_GLINER2_MODEL_ID`, its tests, and the documented example to
   `fastino/gliner2-privacy-filter-PII-multi`; add its model adapter; and adapt the deterministic regex layer from
   Anonymizer PR 265.
2. Implement the deep `TabularPiiReplacer` module, DAG compiler, scopes, deterministic mapping keys, group conflict
   handling, and programmatic entity generators.
3. Implement span normalization, cross-source overlap resolution, component-map reuse, and one-pass text construction.
4. Integrate the replacer after holdout, return the expanded result, persist the resolved plan, and populate statistics.
5. Add focused unit tests, pipeline integration tests, and opt-in live-model evaluations.
6. Run `mise run check ::: test` before handoff or PR work.

## Test and acceptance plan

### Interface PR

- Test `FreeTextDetectionConfig` validation.
- Test immutable cell/span contracts and half-open offset validation.
- Test serialization and documentation changes that make `ReplacePiiConfig.llm` planning-only.
- Test result-model extensions.
- Record group mapping identity, stable first-occurrence behavior, and safe warning payloads as executable contracts
  where possible and durable docstrings otherwise.

### Replacement engine

- Test stable DAG ordering, dependency tuples, record and group scopes, duplicate indexes, replaced group keys, null
  preservation, and protected columns.
- Test that group-scoped dependency drift reuses the first replacement in positional row order and reports one
  aggregate warning/count without raw values.
- Test deterministic seeds, managed fallback, supported generators, patterns, email-domain behavior, organization
  normalization, masks, birth dates, Luhn cards, original inequality, and collision retries.
- Test component replacement reuse only for independently detected spans.
- Test generation elapsed time and distinct-generation counts after cache reuse.

### Detector pipeline

- Test fake GLiNER output and real regex fixtures: lazy loading, label normalization, batching, chunk offsets, overlap
  deduplication, source conflicts, failures, and supported entity coverage.
- Test Anonymizer-compatible longer-span-first resolution, exact ties, touching spans, nested spans, duplicate spans,
  and stable output order.
- Test repeated identical strings and prove that only the occurrences with accepted spans change.
- Test that equal accepted entity/value pairs reuse one replacement across planned free-text columns within a row or
  group.
- Test that structured values, aliases, and address components are not found through string search.
- Confirm no detector is initialized when the plan has no free-text targets.
- Keep real-GLiNER comparisons outside deterministic CI.

### Pipeline acceptance

- Automatic plan discovery sees the full input dataset.
- Only the training split is transformed.
- Original training and test frames remain available to evaluation.
- PII replay consumes complete `ColumnStatistics`, including planned columns with no matches.
- The resolved plan is written to the normal run artifact directory.
- Output shape and index match the input, protected values remain unchanged, and no unplanned columns change.
- No raw PII, detector content, dependency values, or replacement map appears in logs or error messages.

## Open decisions

- Whether built-in regex always runs or the interface exposes a disable switch.
- Whether MAC address and URL join the v3 entity catalog.
- Whether callers need an in-memory complete replacement map and whether secure persistence is ever required.

## Out of scope

- Heuristic plan discovery.
- LLM-based free-text entity detection in the initial implementation.
- LLM-authored replacement values or an LLM sampler backend.
- String-search propagation or replacement of undetected occurrences.
- A propagation-only free-text mode.
- Regex-only continuation after GLiNER failure.
- Custom user-authored regex configuration unless separately approved.
- Compatibility with Nina's obsolete configuration or module structure.

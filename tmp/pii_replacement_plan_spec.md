# PII replacement plan — config shape spec (draft)

Working notes for the v3 plan shape. Implementation may change freely to match this;
the old persona-grouped plan is not a compatibility target.

Reference files:

- Config draft (option A — `ColumnKind` + capabilities):
  `src/nemo_safe_synthesizer/config/replace_pii.py`
- Config draft (option B — dual enums, identify-only on `PiiEntity`):
  `src/nemo_safe_synthesizer/config/replace_pii_alternative.py`
- Option C (dual plan enums, discovery separate): sketch in open questions only
- Example plan: `tmp/example_pii_plan.yaml`

---

## Settled

### Overall shape

- Keep **scope** (`record` | `group` | `dataframe`): how widely an original→synthetic
  mapping stays consistent.
- Drop **persona grouping**. No `persona_backed_columns` / `PersonaColumnSet`.
- One flat list: **`columns_to_replace`**.
- Person-like consistency is expressed only via **`depends_on`** edges, which must
  form a **DAG** (engine topo-sorts before replacement).

### Per-column entry (`PiiColumnPlan`)

| Field | Meaning |
| --- | --- |
| `column_name` | DataFrame column to replace or (for `free_text`) scan/propagate into |
| `entity_type` | **Required.** Replaceable or `free_text` (identify-only refused); see open question on vocabulary encoding |

| `pattern` | Optional **single** format string (strftime / templates). Singular, not a list |
| `depends_on` | Optional list of conditioning columns |

### Conditioning (`depends_on` / `ConditioningColumn`)

Each edge is:

```yaml
depends_on:
  # Read-only conditioner: column_type required
  - column_name: gender
    column_type: sex
  # Replaceable conditioner also in columns_to_replace: omit column_type
  # (inferred from that entry's entity_type)
  - column_name: first_name
```

Two kinds of conditioner:

| Kind | Types | Value used |
| --- | --- | --- |
| Replaceable | `first_name`, `middle_name`, `last_name` | **Synthetic** (replacement) value — so those columns must be replaced first; omit `column_type` and infer from the replace entry |
| Read-only | `sex`, `ethnic_background`, `city`, `state`, `zip_code`, `country`, `organization` | **Original** value; column is not replaced; **`column_type` required** |

Vocabulary note: conditioner type is **`sex`** (not `gender`). The dataframe column
may still be named `gender` / `spouse_gender`; `column_type` is the semantic role.

### Allowed conditioning (entity → allowed `column_type`s)

Only these entity types may declare `depends_on`, and only with the listed types.
Omitting optional conditioners is allowed.

| Entity type | Allowed conditioning columns |
| --- | --- |
| `first_name` | `sex`, `ethnic_background` |
| `middle_name` | `sex`, `ethnic_background` |
| `last_name` | `ethnic_background` |
| `full_name` | `sex`, `ethnic_background` |
| `email` | `first_name`, `middle_name`, `last_name`, `organization` |
| `street_address` | `city`, `state`, `zip_code`, `country` |

All other replaceable entity types (phones, SSN, DOB, IDs, etc.) have **no**
`depends_on`. **`username` / `url` are not first-class entities** (see Free text).

### Name / person coherence (match PGM)

- **No joint `P(first, last)`** and **no identity/group key** in the plan.
- Sample name parts **independently**, each conditioned only on its `depends_on`
  parents (same as `sdg-pgms`: `last_name ← ethnic_background`,
  `first_name ← ethnic_background, sex`; no first↔last edge).
- Cross-field consistency for derived values (email, etc.) comes only from
  explicit `depends_on` edges that consume already-replaced values.
- Multi-person rows (spouse, kids, provider) are just separate columns with their
  own conditioner bindings (e.g. `spouse_gender` vs `gender`); no persona bag.

### Plan validation (rules + where they live)

**Rules to enforce** (all of these; none relaxed for now):

| Rule | Notes |
| --- | --- |
| Allowed `(entity_type → column_type)` matrix | Only entities in the matrix may have `depends_on`; only listed types |
| `free_text` forbids `depends_on` | Already on `PiiColumnPlan` pydantic; keep or mirror in planning validation |
| DAG / acyclicity of `depends_on` | Error should name the cycle |
| No self-edges | Column must not condition on itself |
| Unique `column_name` in `columns_to_replace` | Same as old `pii_plan_duplicate_entry` |
| Unique conditioner `column_type` per column | At most one edge per type on a given replace target |
| Replaceable conditioner refs | `column_name` must appear in `columns_to_replace` with compatible `entity_type` |
| Read-only conditioner refs | Must exist on the dataframe; must **not** be a replace target |
| Columns exist on dataframe | Replace targets and all `depends_on.column_name`s |
| Identify-only / unusable `entity_type` | e.g. `date` (and any other identify-only) cannot be a replace target — same spirit as old validation |
| `scope: group` | Requires `data.group_training_examples_by` set and present on the DF (old behavior; no plan-level `group_by`) |
| Protected columns | Training group / order / timestamp columns must not be replaced (old `protected_columns` checks) |

**Where (mimic old alt-config branch — verified):**

- **Library:** `pii_replacer/planning/validation.py`
- **Preflight:** `preflight/checks/pii.py` → `PiiPlanValidityCheck` (`pii.plan_validity`)
- **Config pydantic (`replace_pii.py`):** keep cheap shape checks that need no dataframe
  (e.g. free_text + `depends_on`). Do **not** put DAG / matrix / DF-ref checks only
  in pydantic — preflight and apply share `iter_plan_issues`.


### Free text

- `entity_type: free_text` means: do **not** replace the cell as one structured
  value; **propagate** already-replaced values into the text.
- **`depends_on` is forbidden** on `free_text` (validated in config).
- **`username` and `url`:** not first-class plan entities. Plan them as
  `free_text` and handle via **propagation** (same as other free text).
  Discovery/heuristics may still **specialize** free-text handling when the
  column name or cell content looks like a username or URL (override the
  generic free-text path), but the plan entity type remains `free_text` and
  still must not use `depends_on`.

### Full address cells planned as `street_address`

- A column that holds a **full address** may still be planned with
  `entity_type: street_address`.
- **Special handling:** replace only the street-line portion; leave city / state /
  zip / country (and any other non-street text) alone.
- No separate `full_address` entity is required for this case.

### `scope: group`

- Keep old-branch behavior: inherit **`data.group_training_examples_by`** as the
  group key. No plan-level `group_by` field.
- Validation: `scope: group` requires that key to be set and present on the DF.

### Sampler config naming (renames)

| Was | Now |
| --- | --- |
| YAML / field `person` | **`sampler`** (short for synthetic value sampler) |
| `PiiPersonConfig` | **`PiiSamplerConfig`** |
| `PiiPersonBackend` | **`PiiSamplerBackend`** (`managed` \| `faker`) |
| Copy like “persona sampler” / “person sampler” | “synthetic value sampler backend…” |

YAML plan fields stay `entity_type` / `depends_on[].column_type` regardless of
which vocabulary option we pick below.

### Illustrative multi-person plan

See `tmp/example_pii_plan.yaml`: primary + spouse names (different `sex` columns,
shared ethnicity), independent phones/DOBs, email conditioned on replaced names,
kids’ `full_name`s conditioned on ethnicity, free-text notes, street / full-address
columns as `street_address`.

---

## Open questions

### 1. Entity / conditioner vocabulary encoding

How to model the overlapping labels (replaceables, identify-only, conditioners)
in the config types. YAML string values stay the same either way; this is about
Python shape and validation.

**Option A — single vocabulary + capabilities**  
File: `src/nemo_safe_synthesizer/config/replace_pii.py`

- One enum: `ColumnKind`
- Roles in `COLUMN_CAPABILITIES` (`replace`, `propagate`, `identify`,
  `condition_synthetic`, `condition_original`)
- Helpers: `can_be_entity_type`, `can_condition`, …
- Matrix: `ALLOWED_DEPENDS_ON`
- Identify-only kinds invalid as `entity_type`; valid as `depends_on.column_type`
  when they have `condition_original`

**Option B — dual enums (identify-only on `PiiEntity`)**  
File: `src/nemo_safe_synthesizer/config/replace_pii_alternative.py`

- `PiiEntity`: replaceables + `free_text` + identify-only (plan naming an
  identify-only type is an error)
- `ConditioningColumnType`: synthetic conditioners (`first_name`, …) + read-only
  conditioners (`sex`, `city`, …)
- Overlapping string values across the two enums; no shared capability map

**Option C — dual plan enums, discovery separate** (sketch only; no `.py` draft)

- `PiiEntity`: **only** values valid as `entity_type` (replaceables + `free_text`).
  Identify-only types (`date`, `sex`, `city`, …) are **not** on this enum, so
  pydantic rejects them as replace targets without a special-case check.
- `ConditioningColumnType`: only values valid as `depends_on.column_type`
  (synthetic name parts + read-only demographics/geo/org).
- Discovery / NER keeps its **own** classification labels (including identify-only)
  and maps them to “exclude from plan” / “candidate conditioner” — not via
  `PiiEntity`.
- Cleanest plan surface; cost is a third vocabulary (or shared string constants)
  for discovery unless it imports conditioner/identify constants from elsewhere.

**Q:** Pick A, B, or C as canonical. A and B have drafts on disk; C would be
implemented if chosen.

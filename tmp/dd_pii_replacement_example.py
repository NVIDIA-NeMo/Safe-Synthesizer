# /// script
# dependencies = [
#   "data-designer",
#   "pydantic",
# ]
# ///
"""Sketch: approximate Safe Synthesizer PII replacement with Data Designer.

The original table is the **seed**. Data Designer already builds a DAG, so the
new `depends_on` plan is a close cousin of `required_columns` / Jinja refs /
`conditional_params`.

What maps well (especially to the *old* persona-backed branch):
  - One managed ``person`` sample ≈ one synthetic identity (first/last/email/
    phone/DOB/national_id drawn together).
  - Multiple people on a row = multiple person sampler columns (patient, spouse,
    provider, kids).
  - Sex conditioning ≈ ``conditional_params`` on the person sampler.
  - Derived strings ≈ ``ExpressionColumnConfig``.
  - Unique IDs ≈ UUID sampler.

What does *not* fall out of the box:
  - Per-row ``ethnic_background`` (person params are batch filters, not Jinja).
  - Independent first/last draws (PGM / new plan). A person sample is joint.
  - Replacement **scope** (record / group / dataframe mapping reuse) — see
    "Honoring scope" below.
  - Free-text **propagation** and street-only replace inside a full address.
  - Overwriting seed columns that already have those names (unique names only).

This script therefore: seed the original row → sample helper person objects
(dropped) → stub a few custom steps → SchemaTransform back to the original
schema. Stubs are intentionally incomplete.

Honoring scope
--------------
Most robust: **post-processing after the full table exists** (Data Designer
``process_after_generation``, not a DAG ``full_column`` which only sees one
generation batch). Draw synthetics first, then rewrite with a map

    (scope_key, column_or_entity, original_value) → synthetic

- ``record``: no map needed if each identity is one person object on the row.
- ``group``: ``scope_key`` = ``group_training_examples_by`` (e.g. patient_id).
- ``dataframe``: omit ``scope_key`` (global). Duplicates of the same original
  (e.g. a provider seen twice) collapse automatically.

Do **not** put this in ``full_column`` / per-batch processors if groups or
repeated values can span batches — you will assign two synthetics to one
original.

Alternative that gets complex: **person-id two-stage join** (collapse unique
identities → sample one person → join back). It works only for columns that
are constant on that grain. It falls apart once a table mixes entities:

- Within a patient_id group, ``provider_name`` is not unique: one patient sees
  several providers, some more than once. Grain must be
  ``(patient_id, provider_name)`` for ``scope: group``, not ``patient_id``.
- The same provider across patients with ``scope: dataframe`` needs a *global*
  ``provider_name`` key, which contradicts the group-scoped patient grain.

So you would need **one collapse/sample/join per entity × scope**, with
different unique keys. That is the same map as post-processing, spread across
several pipelines, plus batch-boundary bugs if you try to fake it with sorted
seed batches. Prefer after-generation remapping.
"""

from __future__ import annotations

from pathlib import Path

import data_designer.config as dd
from pydantic import BaseModel, Field

_HERE = Path(__file__).resolve().parent
_SEED = _HERE / "example_pii_seed.csv"


# ---------------------------------------------------------------------------
# Stubs (new functions the PII replacer would need on top of Data Designer)
# ---------------------------------------------------------------------------


class PersonDrawParams(BaseModel):
    """How to condition a managed-person draw on seed demographics."""

    sex_column: str | None = None
    ethnicity_column: str | None = None
    locale: str = "en_US"


@dd.custom_column_generator(required_columns=["gender", "ethnicity"])
def sample_patient_like_pgm(row: dict, generator_params: PersonDrawParams) -> dict:
    """STUB: draw names independently given sex + ethnic_background (new plan / PGM).

    Data Designer's built-in ``person`` sampler cannot take per-row ethnicity.
    A real implementation would call the managed-assets people generator with
    kwargs derived from ``row[generator_params.sex_column]`` etc., or sample
    first/last from those CPDs separately (no joint first×last table).

    For the *old* persona branch, skip this and use SamplerColumnConfig(person).
    """
    # Placeholder identity so the DAG still type-checks.
    row["patient"] = {
        "first_name": "SYN_FIRST",
        "last_name": "SYN_LAST",
        "sex": row.get("gender"),
        "ethnic_background": row.get("ethnicity"),
        "email_address": "syn.first@example.com",
        "phone_number": "+1-555-0100",
        "national_id": "000-00-0000",
        "birth_date": "1990-01-01",
        "street_number": 1,
        "street_name": "Main",
        "city": "Springfield",
        "region": "IL",
        "postcode": "62701",
        "country": "USA",
    }
    return row


@dd.custom_column_generator(
    required_columns=["first_name", "last_name", "notes_about_family", "username", "patient"]
)
def propagate_and_partial_replace(row: dict) -> dict:
    """STUB: free-text propagation + street-only replace in a full-address cell.

    Walk ``notes_about_family`` / ``username`` and substitute original PII
    strings with the synthetic ones already sampled (no ``depends_on`` on
    free_text — same as the new plan). For ``current_full_address``, replace
    only the street line; leave city/state/zip/country text alone.
    """
    patient = row["patient"]
    notes = str(row.get("notes_about_family") or "")
    notes = notes.replace(str(row.get("first_name") or ""), patient["first_name"])
    notes = notes.replace(str(row.get("last_name") or ""), patient["last_name"])
    row["syn_notes_about_family"] = notes
    return row


class ScopeParams(BaseModel):
    scope: str = Field(description="record | group | dataframe")
    group_key: str | None = None


@dd.custom_column_generator(
    required_columns=["unique_id"],
    # If this stays a column generator, use full_column — but it still only
    # sees one generation batch. Real dataframe/group scope belongs in
    # process_after_generation (see module docstring "Honoring scope").
)
def apply_replacement_scope(df, generator_params: ScopeParams):
    """STUB: remap synthetics to honor replace_pii scope.

    Preferred home: after-generation **post-processing** on the full dataset.

    record: identity-per-row already (ORDERED seed + one person object).
    group: map key (group_id, entity, original) — e.g. same provider_name
        twice for one patient_id reuses one synthetic; a different patient
        may get another.
    dataframe: map key (entity, original) — same provider_name anywhere
        reuses one synthetic.

    Person-id two-stage join is a special case of this map and gets messy
    when one group has multiple providers (and duplicates) or when the same
    provider appears across patients under dataframe scope.
    """
    return df


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    # Original multi-person table. ORDERED ≈ one generated row per seed row
    # (record-scoped replacement without a reuse map).
    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(_SEED)),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    # --- Old-branch style: one person object per identity ----------------
    # Built-in person sampler: joint identity (closer to persona bags than
    # independent first/last). Sex can be gated with conditional_params;
    # ethnicity cannot (use sample_patient_like_pgm instead).

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="patient",
            sampler_type="person",
            drop=True,
            params=dd.PersonSamplerParams(locale="en_US"),
            conditional_params={
                "gender in ['Female', 'F', 'female']": dd.PersonSamplerParams(
                    locale="en_US", sex="Female"
                ),
                "gender in ['Male', 'M', 'male']": dd.PersonSamplerParams(
                    locale="en_US", sex="Male"
                ),
            },
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="spouse",
            sampler_type="person",
            drop=True,
            params=dd.PersonSamplerParams(locale="en_US"),
            conditional_params={
                "spouse_gender in ['Female', 'F', 'female']": dd.PersonSamplerParams(
                    locale="en_US", sex="Female"
                ),
                "spouse_gender in ['Male', 'M', 'male']": dd.PersonSamplerParams(
                    locale="en_US", sex="Male"
                ),
            },
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="provider",
            sampler_type="person",
            drop=True,
            params=dd.PersonSamplerParams(locale="en_US"),
        )
    )
    for kid in ("kid_1", "kid_2", "kid_3"):
        config_builder.add_column(
            dd.SamplerColumnConfig(
                name=kid,
                sampler_type="person",
                drop=True,
                params=dd.PersonSamplerParams(locale="en_US"),
                # STUB: would filter select_field_values={"ethnic_background": [row ethnicity]}
            )
        )

    # Uncomment to prefer PGM-style independent names over a joint person draw:
    # config_builder.add_column(
    #     dd.CustomColumnConfig(
    #         name="patient",
    #         drop=True,
    #         generator_function=sample_patient_like_pgm,
    #         generator_params=PersonDrawParams(
    #             sex_column="gender", ethnicity_column="ethnicity"
    #         ),
    #     )
    # )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="syn_unique_id",
            sampler_type="uuid",
            params=dd.UUIDSamplerParams(prefix="syn-"),
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="syn_spouse_dob",
            expr="{{ spouse.birth_date }}",  # STUB: reformat to %d.%m.%y
            dtype="str",
        )
    )
    config_builder.add_column(
        dd.CustomColumnConfig(
            name="syn_notes_about_family",
            generator_function=propagate_and_partial_replace,
        )
    )
    # Username/URL: same free-text propagation idea; specialize by column name/content.
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="syn_username",
            expr="{{ syn_notes_about_family }}",  # STUB
            dtype="str",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="syn_birth_street_address",
            expr="{{ patient.street_number }} {{ patient.street_name }}",
            dtype="str",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="syn_current_full_address",
            expr="{{ syn_birth_street_address }}",  # STUB: splice street into current_full_address
            dtype="str",
        )
    )

    # Emit a table with the *original* column names, values replaced.
    # SchemaTransform writes a sidecar dataset; seed columns stay on the main
    # frame because Data Designer forbids duplicate column names.
    config_builder.add_processor(
        dd.SchemaTransformProcessorConfig(
            name="replaced_table",
            template={
                "first_name": "{{ patient.first_name }}",
                "last_name": "{{ patient.last_name }}",
                "gender": "{{ gender }}",
                "ethnicity": "{{ ethnicity }}",
                "spouse_first_name": "{{ spouse.first_name }}",
                "spouse_last_name": "{{ spouse.last_name }}",
                "spouse_gender": "{{ spouse_gender }}",
                "phone_number": "{{ patient.phone_number }}",
                "spouse_phone_number": "{{ spouse.phone_number }}",
                "email": "{{ patient.email_address }}",
                "notes_about_family": "{{ syn_notes_about_family }}",
                "username": "{{ syn_username }}",
                "provider_name": "{{ provider.first_name }} {{ provider.last_name }}",
                "kid_full_name_1": "{{ kid_1.first_name }} {{ kid_1.last_name }}",
                "kid_full_name_2": "{{ kid_2.first_name }} {{ kid_2.last_name }}",
                "kid_full_name_3": "{{ kid_3.first_name }} {{ kid_3.last_name }}",
                "ssn": "{{ patient.national_id }}",
                "unique_id": "{{ syn_unique_id }}",
                "date_of_birth": "{{ patient.birth_date }}",
                "spouse_date_of_birth": "{{ syn_spouse_dob }}",
                "birth_street_address": "{{ syn_birth_street_address }}",
                "current_full_address": "{{ syn_current_full_address }}",
            },
        )
    )

    return config_builder

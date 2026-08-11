# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Free-text substitution, LLM stacking seams, and note propagation."""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementScope,
    PiiReplacerConfig,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.entities import config_from_replace_pii
from nemo_safe_synthesizer.pii_replacer.replacer import (
    TabularPiiReplacer,
)
from tests.pii_replacer.helpers import FIRSTS, LASTS


def test_a_surname_mentioned_in_a_note_follows_its_column():
    """The column was replaced but the note kept the real surname, which is the leak."""
    rows = []
    for i in range(60):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        rows.append(
            {
                "patient_name": f"{last.upper()}, {first}",
                "notes": f"Patient {last.upper()} was seen in clinic today for a routine follow up visit {i}",
            }
        )
    df = pd.DataFrame(rows)
    cfg = PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker))
    replacer = TabularPiiReplacer(cfg, data_config=DataParameters())
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df

    for new_name, new_note, original in zip(out["patient_name"], out["notes"], df["patient_name"]):
        new_last, original_last = new_name.split(",")[0], original.split(",")[0]
        assert new_last in new_note, (new_name, new_note)
        if new_last != original_last:
            assert original_last not in new_note, (original, new_note)


def test_llm_enhancement_not_implemented(fixture_patient_df: pd.DataFrame):
    """Defense in depth: apply still refuses if config somehow carries the flag."""
    replacer = TabularPiiReplacer(
        PiiReplacerConfig.model_construct(llm_enhancement=True, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    with pytest.raises(ParameterError, match=r"replace_pii\.llm_enhancement"):
        replacer.transform_df(fixture_patient_df)


def test_noop_enhancer_leaves_heuristics_authoritative(fixture_patient_df: pd.DataFrame):
    from nemo_safe_synthesizer.pii_replacer.llm import NoopEnhancer
    from nemo_safe_synthesizer.pii_replacer.models import DiscoveryResult, PersonaInstance

    cfg = config_from_replace_pii(PiiReplacerConfig(person=PiiPersonConfig(backend=PiiPersonBackend.faker)))
    enhancer = NoopEnhancer()

    discovery = DiscoveryResult()
    assert enhancer.review_discovery(fixture_patient_df, discovery, cfg) is discovery

    instances = [
        PersonaInstance(
            persona="patient",
            match=("record", {"first_name": "Alice"}),
            field_cols={"first_name": "first_name"},
            patterns_by_label={},
            originals={"first_name": "Alice"},
            sex="Female",
        )
    ]
    assert enhancer.infer_persona_demographics(fixture_patient_df, instances, cfg) is instances

    plan = PiiReplacementPlan()
    assert enhancer.detect_freetext_entities(fixture_patient_df, ["notes"], plan, cfg) == []


def test_hand_plan_llm_enhancement_raises_at_freetext_hook():
    """Hand plans skip discovery; apply-time LLM hooks still refuse LLM mode."""
    plan = PiiReplacementPlan(
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person_1",
                columns_to_replace=[PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name)],
            )
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    df = pd.DataFrame({"first_name": ["Alice", "Bob"], "notes": ["saw Alice", "saw Bob"]})
    # Bypass construction validation to exercise the apply-time enhancer seam.
    replacer = TabularPiiReplacer(
        PiiReplacerConfig.model_construct(
            replacement_plan=plan,
            llm_enhancement=True,
            person=PiiPersonConfig(backend=PiiPersonBackend.faker),
        ),
        data_config=DataParameters(),
    )
    with pytest.raises(ParameterError, match=r"replace_pii\.llm_enhancement"):
        replacer.transform_df(df)


def test_record_scoped_free_text_follows_each_rows_persona():
    """Duplicate structured names under record scope must not merge free-text pairs."""
    df = pd.DataFrame(
        {
            "first_name": ["Alice", "Alice"],
            "notes": ["Alice visited the clinic", "Alice called the nurse"],
        }
    )
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.record,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            )
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df
    names = out["first_name"].tolist()
    notes = out["notes"].tolist()
    assert names[0] != names[1]
    assert names[0] in notes[0] and "Alice" not in notes[0]
    assert names[1] in notes[1] and "Alice" not in notes[1]
    assert names[0] not in notes[1]
    assert names[1] not in notes[0]


def test_a_name_token_is_aliased_without_the_punctuation_around_it():
    """Free text writes 'SMITH', never 'SMITH,', so the comma must not travel with the token."""
    from nemo_safe_synthesizer.pii_replacer.replacement import instance_text_pairs

    inst = {"originals": {"full_name": "SMITH, Jane"}, "synthetic": {"full_name": "JONES, Robert"}}

    pairs = dict(instance_text_pairs(inst))
    assert pairs["SMITH"] == "JONES"
    assert pairs["Jane"] == "Robert"


def test_a_name_keeps_the_punctuation_inside_it():
    from nemo_safe_synthesizer.pii_replacer.replacement import instance_text_pairs

    inst = {"originals": {"full_name": "Jane O'Brien"}, "synthetic": {"full_name": "Robert Smith-Jones"}}

    assert dict(instance_text_pairs(inst))["O'Brien"] == "Smith-Jones"


def test_an_aliased_token_is_labelled_by_its_role():
    """Which token is the surname is the column's business, not the token's position."""
    from nemo_safe_synthesizer.pii_replacer.replacement import instance_text_pair_labels

    inst = {"originals": {"full_name": "SMITH, Jane"}, "synthetic": {"full_name": "JONES, Robert"}}

    labels = instance_text_pair_labels(inst)
    assert labels["SMITH"] == "last_name"
    assert labels["Jane"] == "first_name"


def test_free_text_substitution_is_case_insensitive():
    from nemo_safe_synthesizer.pii_replacer.replacement import build_text_substituter

    sub = build_text_substituter([("Smith", "Jones")])
    assert sub is not None
    assert sub("Patient SMITH was seen") == "Patient JONES was seen"
    assert sub("patient smith was seen") == "patient jones was seen"
    assert sub("Patient Smith was seen") == "Patient Jones was seen"


def test_group_free_text_uses_row_local_persona_pairs(fixture_patient_df: pd.DataFrame):
    """Provider pairs from one row must not rewrite another row's notes in the same group."""
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.group,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            ),
            PersonaColumnSet(
                persona="provider",
                columns_to_replace=[
                    PiiColumnPlan(column_name="provider_name", entity_type=PiiEntity.full_name),
                ],
            ),
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(fixture_patient_df)
    assert replacer.result is not None
    out = replacer.result.transformed_df
    # Row 0 mentions Dr X; row 1 mentions Dr Y. After replacement each note should
    # contain that row's provider synthetic, not the other row's.
    row0_provider = str(out.loc[0, "provider_name"])
    row1_provider = str(out.loc[1, "provider_name"])
    note0 = str(out.loc[0, "notes"])
    note1 = str(out.loc[1, "notes"])
    assert row0_provider != row1_provider
    assert row0_provider in note0
    assert row1_provider in note1
    assert row0_provider not in note1
    assert row1_provider not in note0
    assert "Dr X" not in note0 and "Dr Y" not in note1


def test_standalone_values_propagate_into_free_text():
    """Replaced standalone IDs mentioned in notes are rewritten to match the column."""
    n = 20
    event_ids = [f"EVT-{i:05d}" for i in range(n)]
    df = pd.DataFrame(
        {
            "event_id": event_ids,
            "notes": [
                f"Ticket {eid} opened after patient visited clinic for follow up care today" for eid in event_ids
            ],
        }
    )
    plan = PiiReplacementPlan(
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="event_id", entity_type=PiiEntity.unique_identifier),
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    replacer = TabularPiiReplacer(
        PiiReplacerConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(),
    )
    replacer.transform_df(df)
    assert replacer.result is not None
    out = replacer.result.transformed_df
    for i in range(n):
        orig = event_ids[i]
        new_id = out["event_id"].iloc[i]
        note = out["notes"].iloc[i]
        assert new_id != orig
        assert orig not in note
        assert new_id in note

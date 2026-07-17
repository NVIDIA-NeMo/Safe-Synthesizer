# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.pii_replacement import (
    PiiColumnPlan,
    PiiEntity,
    PiiPersona,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer import (
    PiiPlanPreview,
    TabularPiiReplacer,
    build_plan_preview,
    render_plan_preview_html,
)


@pytest.fixture
def preview_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p-1", "p-2"],
            "first_name": ["Alice <Admin>", "Bob"],
            "sex": ["Female", "Male"],
            "doctor_name": ["Dr. One", "Dr. Two"],
            "notes": ["Alice called Dr. One", "No PII found"],
            "event_type": ["visit", "call"],
        }
    )


@pytest.fixture
def replacement_plan() -> PiiReplacementPlan:
    return PiiReplacementPlan(
        group_key="patient_id",
        identified_personas={
            "patient": PiiPersona(gender="sex"),
            "doctor": None,
        },
        columns={
            "patient_id": PiiColumnPlan(entity_type=PiiEntity.unique_identifier),
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="patient"),
            "doctor_name": PiiColumnPlan(entity_type=PiiEntity.full_name, persona="doctor"),
            "notes": PiiColumnPlan(entity_type=PiiEntity.free_text),
        },
    )


def test_render_plan_preview_shows_scope_entities_and_related_groups(
    preview_df: pd.DataFrame,
    replacement_plan: PiiReplacementPlan,
) -> None:
    rendered = render_plan_preview_html(replacement_plan, preview_df.head(1), total_rows=len(preview_df))

    assert "PII Replacement Plan Preview" in rendered
    assert "Detection and plan only" in rendered
    assert "Entire column" in rendered
    assert "Matching values only" in rendered
    assert "first_name" in rendered
    assert "free_text" in rendered
    assert "Related Persona Groups" in rendered
    assert "patient" in rendered
    assert "doctor" in rendered
    assert "Gender: sex" in rendered
    assert "Dataset group key" in rendered
    assert "FULL" in rendered
    assert "MATCHES" in rendered


def test_render_plan_preview_escapes_original_values_and_plan_labels(preview_df: pd.DataFrame) -> None:
    plan = PiiReplacementPlan(
        identified_personas={"patient<script>": None},
        columns={
            "first_name": PiiColumnPlan(entity_type=PiiEntity.first_name, persona="patient<script>"),
        },
    )

    rendered = render_plan_preview_html(plan, preview_df.head(1))

    assert "Alice &lt;Admin&gt;" in rendered
    assert "patient&lt;script&gt;" in rendered
    assert "Alice <Admin>" not in rendered
    assert "patient<script>" not in rendered


def test_render_plan_preview_shows_mixed_scope_and_sample_values(preview_df: pd.DataFrame) -> None:
    plan = PiiReplacementPlan(
        columns={
            "patient_id": PiiColumnPlan(
                entity_type=PiiEntity.unique_identifier,
                pattern="pmc-*",
                dominant_pattern_coverage=60.0,
            ),
        }
    )
    rendered = render_plan_preview_html(
        plan,
        preview_df.head(1),
        column_stats={"patient_id": {"samples": ["p-1", "p-2"], "scope": "key"}},
        dominant_pattern_min_coverage=85.0,
    )

    assert "Mixed values" in rendered
    assert "MIXED" in rendered
    assert "Sample values" in rendered
    assert "p-1, p-2" in rendered
    assert "Column scope" in rendered
    assert ">key<" in rendered


def test_build_plan_preview_resolves_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
    preview_df: pd.DataFrame,
    replacement_plan: PiiReplacementPlan,
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("replacement must not run while previewing the plan")

    monkeypatch.setattr("nemo_safe_synthesizer.pii_replacer.replacer.run_replacement", fail_if_called)
    preview = build_plan_preview(
        preview_df,
        ReplacePiiConfig(replacement_plan=replacement_plan),
        data_config=DataParameters(group_training_examples_by="patient_id"),
        num_rows=1,
    )

    assert isinstance(preview, PiiPlanPreview)
    assert preview.plan == replacement_plan
    assert preview.column_stats["patient_id"]["scope"] == "key"
    assert preview.sample_dataframe.equals(preview_df.head(1))


def test_preview_plan_resolves_without_running_replacement(
    monkeypatch: pytest.MonkeyPatch,
    preview_df: pd.DataFrame,
    replacement_plan: PiiReplacementPlan,
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("replacement must not run while previewing the plan")

    monkeypatch.setattr("nemo_safe_synthesizer.pii_replacer.replacer.run_replacement", fail_if_called)
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=replacement_plan),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )

    preview = replacer.preview_plan(preview_df, num_rows=1)

    assert isinstance(preview, PiiPlanPreview)
    assert preview.total_rows == 2
    assert preview.sample_dataframe.equals(preview_df.head(1))
    assert replacer.resolved_plan == replacement_plan
    assert replacer.result is None
    assert preview._repr_html_() == preview.to_html()


def test_preview_plan_can_omit_unaffected_sample_columns(
    preview_df: pd.DataFrame,
    replacement_plan: PiiReplacementPlan,
) -> None:
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=replacement_plan),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )

    rendered = replacer.preview_plan(preview_df, include_unaffected=False).to_html()

    assert "event_type" not in rendered
    assert "Unaffected context" not in rendered
    assert "sex" in rendered
    assert "CONDITION" in rendered


def test_preview_plan_rejects_negative_row_count(
    preview_df: pd.DataFrame,
    replacement_plan: PiiReplacementPlan,
) -> None:
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=replacement_plan),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )

    with pytest.raises(ValueError, match="num_rows must be non-negative"):
        replacer.preview_plan(preview_df, num_rows=-1)

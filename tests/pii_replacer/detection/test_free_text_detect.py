# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which columns are read as prose, and when they are read at all."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.pii_replacement import (
    PiiEntity,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.detection import select_free_text_columns
from nemo_safe_synthesizer.pii_replacer.entities import config_from_replace_pii
from tests.pii_replacer.helpers import column_spec


def test_free_text_excludes_non_object_and_structured_field_types():
    df = pd.DataFrame(
        {
            "weight": [135.0, 165.0, 142.0],
            "is_active": [True, False, True],
            "event_type": ["Admission", "Admission", "Discharge"],
            "notes": [
                "Patient visited clinic for follow up care today",
                "Another long clinical note about symptoms and treatment",
                "Third detailed note with multiple words in the sentence",
            ],
        }
    )
    free_text = select_free_text_columns(df, exclude=set())
    assert free_text == ["notes"]


def test_nss_free_text_detection_matches_describe_field():
    df = pd.DataFrame(
        {
            "code": list(range(20)),
            "tag": [f"tag{i:02d}" for i in range(20)],
            "notes": [f"Patient record {i} visited clinic for follow up care and discussion today" for i in range(20)],
        }
    )
    assert set(select_free_text_columns(df, set())) == {"notes", "tag"}


def test_non_llm_mode_skips_free_text_scan_without_structured_columns(caplog):
    import logging

    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    caplog.set_level(logging.INFO)
    # No persona columns and no replaceable standalone columns: only a generic
    # (identify-only) date and a free-text column. Without structured PII there
    # is nothing to propagate, so the free-text column must not be scanned/planned.
    dominant_dates = [f"04/{(i % 28) + 1:02d}/2023" for i in range(100)]
    df = pd.DataFrame(
        {
            "event_date": dominant_dates,
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(100)],
        }
    )
    plan = discover_plan(
        df,
        group_key=None,
        cfg=config_from_replace_pii(ReplacePiiConfig()),
        config=ReplacePiiConfig(),
    )
    assert column_spec(plan.standalone_columns_to_replace, "notes") is None
    for col_set in plan.persona_backed_columns:
        assert column_spec(col_set.columns_to_replace, "notes") is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("will not be scanned or replaced" in message for message in messages)


def test_free_text_planned_with_standalone_only_structured_columns():
    """Standalone entity columns alone are enough to enable free-text scanning."""
    from nemo_safe_synthesizer.pii_replacer.planning import discover_plan

    n = 40
    df = pd.DataFrame(
        {
            "event_id": [f"EVT-{i:05d}" for i in range(n)],
            "notes": [f"Patient record {i} visited clinic for follow up care today" for i in range(n)],
        }
    )
    plan = discover_plan(df, None, config_from_replace_pii(ReplacePiiConfig()), ReplacePiiConfig())
    assert plan.persona_backed_columns == []
    assert any(s.entity_type == PiiEntity.unique_identifier for s in plan.standalone_columns_to_replace)
    notes = column_spec(plan.standalone_columns_to_replace, "notes")
    assert notes is not None and notes.entity_type == PiiEntity.free_text

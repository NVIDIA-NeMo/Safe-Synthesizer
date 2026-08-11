# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics, TransformResult
from nemo_safe_synthesizer.tooling.pii_replacement_result_preview import (
    _DATA_COL_PX,
    _TEXT_COL_PX,
    build_changed_record_indices,
    build_interlaced_comparison_frame,
    preview_pii_replacement_result,
    render_comparison_tables_html,
)


def test_interlaced_frame_and_html_headers():
    original = pd.DataFrame(
        {
            "first_name": ["Aisha", "Ben"],
            "notes": ["Patient Aisha visited", "ok"],
            "event_id": [1, 2],
        }
    )
    transformed = original.copy()
    transformed.loc[0, "first_name"] = "Erica"
    transformed.loc[0, "notes"] = "Patient Erica visited"

    frame = build_interlaced_comparison_frame(
        original.loc[[0]],
        transformed.loc[[0]],
        columns=["first_name", "notes", "event_id"],
    )
    assert list(frame.index.get_level_values("side")) == ["original", "transformed"]
    assert frame.loc[(0, "original"), "first_name"] == "Aisha"
    assert frame.loc[(0, "transformed"), "first_name"] == "Erica"

    stats = {
        "first_name": ColumnStatistics(
            assigned_type="structured",
            assigned_entity="first_name",
            is_transformed=True,
        ),
        "notes": ColumnStatistics(
            assigned_type="text",
            assigned_entity="free_text",
            is_transformed=True,
        ),
    }
    html_out = render_comparison_tables_html(
        original.loc[[0]],
        transformed.loc[[0]],
        column_statistics=stats,
        free_text_entities=[
            {
                "column": "notes",
                "original": "Aisha",
                "synthetic": "Erica",
                "label": "first_name",
            }
        ],
        columns=["first_name", "notes", "event_id"],
    )
    # Headers carry column name and entity type; sanitizers strip <style>, so
    # every rule must be inline.
    assert "<style>" not in html_out
    assert "first_name</div>" in html_out
    assert ">free_text</div>" in html_out
    assert "table-layout:fixed" in html_out
    assert f"width:{_TEXT_COL_PX}px" in html_out
    assert f"width:{_DATA_COL_PX}px" in html_out
    assert html_out.index(">original</div>") < html_out.index(">transformed</div>")
    assert "border-radius:3px" in html_out
    assert "| first_name" not in html_out


def test_changed_indices_prioritize_text_column_changes():
    original = pd.DataFrame(
        {
            "first_name": ["Aisha", "Ben", "Cara"],
            "notes": ["keep", "text Ben here", "keep"],
        }
    )
    transformed = original.copy()
    transformed.loc[0, "first_name"] = "Erica"
    transformed.loc[1, "notes"] = "text Dana here"
    transformed.loc[2, "first_name"] = "Dana"

    stats = {
        "first_name": ColumnStatistics(
            assigned_type="structured",
            assigned_entity="first_name",
            is_transformed=True,
        ),
        "notes": ColumnStatistics(
            assigned_type="text",
            assigned_entity="free_text",
            is_transformed=True,
        ),
    }
    indices = build_changed_record_indices(
        original,
        transformed,
        columns=["first_name", "notes"],
        column_statistics=stats,
        max_records=10,
    )
    assert indices[0] == 1
    assert set(indices) == {0, 1, 2}


def test_preview_result_repr_html():
    original = pd.DataFrame(
        {
            "first_name": ["Aisha", "Ben", "Cara"],
            "city": ["Austin", "Boston", "Chicago"],
        }
    )
    transformed = original.copy()
    transformed.loc[0, "first_name"] = "Erica"
    transformed.loc[2, "first_name"] = "Dana"
    stats = {
        "first_name": ColumnStatistics(
            assigned_type="structured",
            assigned_entity="first_name",
            is_transformed=True,
        )
    }
    result = TransformResult(transformed_df=transformed, column_statistics=stats, free_text_entities=[])
    preview = preview_pii_replacement_result(
        original_df=original,
        transform_result=result,
        max_records=10,
        only_changed=True,
    )
    html_out = preview._repr_html_()
    assert "Erica" in html_out
    assert "Dana" in html_out
    assert "first_name</div>" in html_out
    assert len(preview.interlaced_df) == 4  # 2 records × 2 sides

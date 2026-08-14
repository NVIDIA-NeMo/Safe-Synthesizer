# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

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
    # Column-interlaced: one row per record, (column, side) sub-columns.
    assert list(frame.index) == [0]
    assert list(frame.columns.get_level_values("side")[:2]) == ["original", "transformed"]
    assert frame.loc[0, ("first_name", "original")] == "Aisha"
    assert frame.loc[0, ("first_name", "transformed")] == "Erica"

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


def _chip_colors_by_value(html_out: str) -> dict[str, set[str]]:
    colors: dict[str, set[str]] = {}
    for color, value in re.findall(r"border:1\.5px solid (#[0-9a-f]{6});[^>]*>([^<]+)</span>", html_out):
        colors.setdefault(value, set()).add(color)
    return colors


def test_chip_color_is_per_value_pair_not_per_column():
    original = pd.DataFrame(
        {
            "first_name": ["Aisha", "Ben"],
            "notes": ["Aisha called", "Ben called"],
        }
    )
    transformed = original.copy()
    transformed.loc[0, "first_name"] = "Erica"
    transformed.loc[0, "notes"] = "Erica called"
    transformed.loc[1, "first_name"] = "Dana"
    transformed.loc[1, "notes"] = "Dana called"

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
        original,
        transformed,
        column_statistics=stats,
        free_text_entities=[
            {"column": "notes", "original": "Aisha", "synthetic": "Erica", "label": "first_name"},
            {"column": "notes", "original": "Ben", "synthetic": "Dana", "label": "first_name"},
        ],
        columns=["first_name", "notes"],
    )
    colors = _chip_colors_by_value(html_out)

    # A replaced value and its replacement share one color, in every column it
    # appears in (structured cell and inside free text).
    assert len(colors["Aisha"]) == 1
    assert colors["Aisha"] == colors["Erica"]
    assert colors["Ben"] == colors["Dana"]
    # Distinct pairs get distinct colors, so one column is not one color.
    assert colors["Aisha"] != colors["Ben"]


def test_free_text_chips_respect_word_boundaries_and_changed_cells():
    # Rows 0/1 are untouched; row 2 holds the standalone "Lab" value that put
    # the short needle into free_text_entities in the first place.
    original = pd.DataFrame({"event_name": ["Debridement of eschar", "Laboratory Tests", "Lab"]})
    transformed = pd.DataFrame({"event_name": ["Debridement of eschar", "Laboratory Tests", "Radiology"]})
    stats = {
        "event_name": ColumnStatistics(
            assigned_type="text",
            assigned_entity="free_text",
            is_transformed=True,
        )
    }
    html_out = render_comparison_tables_html(
        original,
        transformed,
        column_statistics=stats,
        free_text_entities=[
            {"column": "event_name", "original": "Lab", "synthetic": "Radiology", "label": "organization"}
        ],
        columns=["event_name"],
    )

    # "Lab" must not chip inside "Laboratory", nor "ent" inside "Debridement":
    # the replacer only substitutes on word boundaries.
    assert "Debridement of eschar" in html_out
    assert "Laboratory Tests" in html_out
    assert ">Lab</span>oratory" not in html_out
    # The real standalone replacement is still chipped on both sides.
    chipped = {value for _, value in re.findall(r"border:1\.5px solid (#[0-9a-f]{6});[^>]*>([^<]+)</span>", html_out)}
    assert chipped == {"Lab", "Radiology"}


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
    assert len(preview.interlaced_df) == 2  # 2 changed records, one row each
    # Column-interlaced: each source column yields an original/transformed pair.
    assert ("first_name", "original") in preview.interlaced_df.columns
    assert ("first_name", "transformed") in preview.interlaced_df.columns

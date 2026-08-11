# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook preview of original vs PII-replaced tabular data.

Builds an interlaced DataFrame (original/transformed row pairs), then renders a
single HTML table with colored chips on replaced values. Entity types appear in
column headers. Prefer :meth:`SafeSynthesizer.preview_replaced_data` after
``process_data()``.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from ..errors import ParameterError
from ..pii_replacer.transform_result import ColumnStatistics, TransformResult

__all__ = [
    "PiiReplacementResultPreview",
    "build_changed_record_indices",
    "build_interlaced_comparison_frame",
    "preview_pii_replacement_result",
    "render_comparison_tables_html",
]

# Match Anonymizer label border palette for chip colors.
_LABEL_BORDER_COLORS: list[str] = [
    "#3b82f6",
    "#22c55e",
    "#f59e0b",
    "#ec4899",
    "#6366f1",
    "#10b981",
    "#f97316",
    "#8b5cf6",
    "#06b6d4",
    "#eab308",
    "#a855f7",
    "#ef4444",
]

_SIDE_ORIGINAL = "original"
_SIDE_TRANSFORMED = "transformed"


def _border_for_label(label: str) -> str:
    return _LABEL_BORDER_COLORS[hash(label) % len(_LABEL_BORDER_COLORS)]


def _chip(value: str, color_key: str) -> str:
    border = _border_for_label(color_key)
    display = value if value else "(empty)"
    return (
        f'<span style="display:inline;border:1.5px solid {html.escape(border)};'
        f'padding:1px 4px;border-radius:3px;background:#fff">{html.escape(display)}</span>'
    )


def _plain(value: str) -> str:
    return html.escape(value)


def _cell_str(value: object) -> str:
    if value is None or value is pd.NaT or value is pd.NA:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _entity_for_column(column_statistics: dict[str, ColumnStatistics], column: str) -> str | None:
    stats = column_statistics.get(column)
    if stats is None:
        return None
    return stats.assigned_entity


def _color_key_for_column(column_statistics: dict[str, ColumnStatistics], column: str) -> str:
    return _entity_for_column(column_statistics, column) or column


def _free_text_pairs_for_column(
    free_text_entities: list[dict[str, Any]],
    column: str,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ent in free_text_entities:
        if ent.get("column") != column:
            continue
        original = str(ent.get("original") or "")
        synthetic = str(ent.get("synthetic") or "")
        if not original:
            continue
        key = (original.lower(), synthetic.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((original, synthetic))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def _highlight_with_pairs(text: str, pairs: list[tuple[str, str]], *, side: str, color_key: str) -> str:
    if not text or not pairs:
        return _plain(text)

    needles = [(original if side == _SIDE_ORIGINAL else synthetic) for original, synthetic in pairs]
    needles = [n for n in needles if n]
    needles.sort(key=len, reverse=True)
    if not needles:
        return _plain(text)

    lower = text.lower()
    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        match_at: int | None = None
        match_len = 0
        for needle in needles:
            idx = lower.find(needle.lower(), cursor)
            if idx < 0:
                continue
            if match_at is None or idx < match_at or (idx == match_at and len(needle) > match_len):
                match_at = idx
                match_len = len(needle)
        if match_at is None:
            parts.append(html.escape(text[cursor:]))
            break
        if match_at > cursor:
            parts.append(html.escape(text[cursor:match_at]))
        parts.append(_chip(text[match_at : match_at + match_len], color_key))
        cursor = match_at + match_len
    return "".join(parts) if parts else _plain(text)


def _render_cell_html(
    column: str,
    value: object,
    *,
    other_value: object,
    column_statistics: dict[str, ColumnStatistics],
    free_text_entities: list[dict[str, Any]],
    side: str,
) -> str:
    text = _cell_str(value)
    other = _cell_str(other_value)
    changed = text != other
    color_key = _color_key_for_column(column_statistics, column)
    stats = column_statistics.get(column)
    is_free_text = bool(stats and stats.assigned_entity == "free_text")

    if is_free_text:
        pairs = _free_text_pairs_for_column(free_text_entities, column)
        if pairs:
            return _highlight_with_pairs(text, pairs, side=side, color_key=color_key)
        if changed:
            return _chip(text, color_key)
        return _plain(text)

    if changed:
        return _chip(text, color_key)
    return _plain(text)


def build_interlaced_comparison_frame(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build an interlaced frame: original row, transformed row, original, ...

    Index is a ``MultiIndex`` of ``(record, side)`` where ``side`` is
    ``original`` or ``transformed``.
    """
    cols = columns or [c for c in original_df.columns if c in transformed_df.columns]
    transformed_aligned = transformed_df.reindex(original_df.index)
    blocks: list[pd.DataFrame] = []
    for idx in original_df.index:
        original_row = original_df.loc[[idx], cols]
        transformed_row = transformed_aligned.loc[[idx], cols]
        pair = pd.concat([original_row, transformed_row], axis=0)
        pair.index = pd.MultiIndex.from_arrays(
            [[idx, idx], [_SIDE_ORIGINAL, _SIDE_TRANSFORMED]],
            names=["record", "side"],
        )
        blocks.append(pair)
    if not blocks:
        return pd.DataFrame(columns=cols)
    return pd.concat(blocks, axis=0)


def _text_columns(column_statistics: dict[str, ColumnStatistics], columns: list[str]) -> list[str]:
    return [
        col
        for col in columns
        if (stats := column_statistics.get(col)) is not None and stats.assigned_entity == "free_text"
    ]


def build_changed_record_indices(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    column_statistics: dict[str, ColumnStatistics] | None = None,
    max_records: int | None = None,
) -> list[Any]:
    """Return index labels for rows where any compared column changed.

    When ``column_statistics`` marks free-text columns, rows with text-column
    changes are listed first.
    """
    cols = columns or [c for c in original_df.columns if c in transformed_df.columns]
    if not cols:
        return []
    orig = original_df.loc[:, cols].map(_cell_str)
    new = transformed_df.loc[:, cols].reindex(original_df.index).map(_cell_str)
    changed = orig != new
    changed_mask = changed.any(axis=1)
    changed_indices = list(original_df.index[changed_mask])

    text_cols = _text_columns(column_statistics or {}, cols)
    if text_cols:
        text_changed = changed.loc[changed_indices, text_cols].any(axis=1)
        text_first = [idx for idx in changed_indices if bool(text_changed.loc[idx])]
        other = [idx for idx in changed_indices if idx not in set(text_first)]
        changed_indices = text_first + other

    if max_records is not None:
        return changed_indices[: max(0, max_records)]
    return changed_indices


_INDEX_COL_PX = 104
_DATA_COL_PX = 190
_TEXT_COL_PX = 380

_CELL_BASE = (
    "border-bottom:1px solid #edf1f5;border-right:1px solid #edf1f5;"
    "padding:6px 9px;text-align:left;vertical-align:top;"
    "white-space:pre-wrap;overflow-wrap:break-word;word-break:normal;"
)
_INDEX_CELL_BASE = (
    "border-bottom:1px solid #edf1f5;border-right:1px solid #c9d2dc;"
    "padding:6px 9px;text-align:left;vertical-align:top;white-space:nowrap;"
    "color:#5b6b7c;font-weight:500;"
)
_HEADER_BG = "background:#f1f5f9;"


def _column_width_px(column: str, column_statistics: dict[str, ColumnStatistics]) -> int:
    stats = column_statistics.get(column)
    if stats is not None and stats.assigned_entity == "free_text":
        return _TEXT_COL_PX
    return _DATA_COL_PX


def _render_header_cell(column: str, column_statistics: dict[str, ColumnStatistics]) -> str:
    entity = _entity_for_column(column_statistics, column)
    border = _border_for_label(_color_key_for_column(column_statistics, column))
    entity_html = (
        f'<div style="margin-top:2px;font-size:11px;font-weight:600;color:{html.escape(border)}">'
        f"{html.escape(entity)}</div>"
        if entity
        else ""
    )
    return (
        f'<th scope="col" style="{_CELL_BASE}{_HEADER_BG}">'
        f'<div style="font-weight:700;color:#1f2a37">{html.escape(column)}</div>'
        f"{entity_html}"
        f"</th>"
    )


def render_comparison_tables_html(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    column_statistics: dict[str, ColumnStatistics],
    free_text_entities: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    title: str = "PII Replacement Result Preview",
) -> str:
    """Render one interlaced Original/Transformed HTML table with chips.

    All styling is inline: notebook HTML sanitizers commonly strip ``<style>``
    blocks, which previously dropped the header styling and column widths.
    """
    free_text_entities = free_text_entities or []
    cols = columns or [c for c in original_df.columns if c in transformed_df.columns]
    interlaced = build_interlaced_comparison_frame(original_df, transformed_df, columns=cols)
    transformed_aligned = transformed_df.reindex(original_df.index)

    widths = [_column_width_px(col, column_statistics) for col in cols]
    total_px = _INDEX_COL_PX + sum(widths)
    colgroup = (
        f'<colgroup><col style="width:{_INDEX_COL_PX}px" />'
        + "".join(f'<col style="width:{width}px" />' for width in widths)
        + "</colgroup>"
    )

    header_cells = [f'<th scope="col" style="{_INDEX_CELL_BASE}{_HEADER_BG}">record</th>'] + [
        _render_header_cell(col, column_statistics) for col in cols
    ]

    body_rows: list[str] = []
    for index, row in interlaced.iterrows():
        record_id, side = cast("tuple[Any, str]", index)  # (record, ORIGINAL/TRANSFORMED) MultiIndex
        other_source = transformed_aligned if side == _SIDE_ORIGINAL else original_df
        other_row = other_source.loc[record_id] if record_id in other_source.index else row
        if isinstance(other_row, pd.DataFrame):
            other_row = other_row.iloc[0]
        is_transformed = side == _SIDE_TRANSFORMED
        row_bg = "#f4faf9" if is_transformed else "#ffffff"
        row_border = "2px solid #d7e3e8" if is_transformed else "1px solid #edf1f5"
        side_color = "#0f6a6a" if is_transformed else "#6b7c8a"
        cell_style = f"{_CELL_BASE}background:{row_bg};border-bottom:{row_border};"
        index_style = f"{_INDEX_CELL_BASE}background:#fafbfc;border-bottom:{row_border};"
        cells = [
            f'<td style="{cell_style}">'
            + _render_cell_html(
                col,
                row.get(col),
                other_value=other_row.get(col) if hasattr(other_row, "get") else other_row[col],
                column_statistics=column_statistics,
                free_text_entities=free_text_entities,
                side=str(side),
            )
            + "</td>"
            for col in cols
        ]
        body_rows.append(
            f"<tr>"
            f'<th scope="row" style="{index_style}">'
            f'<div style="font-weight:700;color:#1f2a37">{html.escape(str(record_id))}</div>'
            f'<div style="margin-top:2px;font-size:10.5px;text-transform:uppercase;'
            f'letter-spacing:0.04em;color:{side_color}">{html.escape(str(side))}</div>'
            f"</th>"
            f"{''.join(cells)}"
            f"</tr>"
        )

    return f"""
<div style="font-family:'IBM Plex Sans','Segoe UI',sans-serif;color:#1f2a37;border:1px solid #c9d2dc;\
background:#f7f9fb;border-radius:8px;padding:12px;max-width:100%">
  <div style="font-size:13px;color:#5b6b7c;margin-bottom:10px">Showing {len(original_df)} record(s) \
&times; {len(cols)} column(s), interlaced original / transformed</div>
  <div style="border:1px solid #c9d2dc;border-radius:8px;overflow:hidden;background:#fff">
    <div style="padding:10px 14px;border-bottom:1px solid #c9d2dc;font-weight:650;background:#f3f6f9">\
{html.escape(title)}</div>
    <div style="padding:12px 14px">
      <div style="overflow:auto;max-height:34rem;border:1px solid #c9d2dc;border-radius:6px">
        <table style="border-collapse:collapse;table-layout:fixed;width:{total_px}px;\
font-family:'IBM Plex Mono','SFMono-Regular',Menlo,Consolas,monospace;font-size:12.5px;line-height:1.45">
          {colgroup}
          <thead><tr>{"".join(header_cells)}</tr></thead>
          <tbody>{"".join(body_rows)}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>
""".strip()


@dataclass
class PiiReplacementResultPreview:
    """Notebook display object for interlaced original/transformed tables."""

    html: str
    interlaced_df: pd.DataFrame
    status: str

    def _repr_html_(self) -> str:
        return self.html


def preview_pii_replacement_result(
    *,
    original_df: pd.DataFrame,
    transform_result: TransformResult,
    max_records: int = 10,
    only_changed: bool = True,
    columns: list[str] | None = None,
) -> PiiReplacementResultPreview:
    """Show original vs transformed tables after PII replacement (``.head()``-style)."""
    transformed_df = transform_result.transformed_df
    stats = transform_result.column_statistics
    focus_cols = columns
    if focus_cols is None:
        changed_cols = [c for c, s in stats.items() if s.is_transformed]
        focus_cols = changed_cols or [c for c in original_df.columns if c in transformed_df.columns]

    if only_changed:
        indices = build_changed_record_indices(
            original_df,
            transformed_df,
            columns=focus_cols,
            column_statistics=stats,
            max_records=max_records,
        )
    else:
        indices = list(original_df.index[:max_records])

    if not indices:
        raise ParameterError("No changed records found to preview after PII replacement")

    sample_original = original_df.loc[indices, focus_cols]
    sample_transformed = transformed_df.reindex(indices).loc[:, focus_cols]
    interlaced = build_interlaced_comparison_frame(sample_original, sample_transformed, columns=focus_cols)
    html_out = render_comparison_tables_html(
        sample_original,
        sample_transformed,
        column_statistics=stats,
        free_text_entities=list(transform_result.free_text_entities),
        columns=focus_cols,
    )
    status = f"Showing {len(sample_original)} record(s) × {len(focus_cols)} column(s)"
    return PiiReplacementResultPreview(html=html_out, interlaced_df=interlaced, status=status)

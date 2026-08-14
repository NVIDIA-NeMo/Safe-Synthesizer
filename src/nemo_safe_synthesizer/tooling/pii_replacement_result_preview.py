# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook preview of original vs PII-replaced tabular data.

Builds a column-interlaced DataFrame: each source column becomes an
``original`` / ``transformed`` sub-column pair sitting side by side, with one
row per record. The single HTML table puts colored chips on replaced values,
where one color is assigned per distinct original/transformed value pair, so a
replaced value and its replacement share a chip color wherever they appear.
Entity types appear in the column headers. Prefer
:meth:`SafeSynthesizer.preview_replaced_data` after ``process_data()``.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

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

# Chip border palette; one entry is assigned to each distinct original/transformed
# value pair, in order of first appearance.
_PAIR_BORDER_COLORS: list[str] = [
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


class _PairColors:
    """Assigns one chip color per distinct original/transformed value pair.

    Colors are allocated on first sight and keyed on the normalized value pair,
    so the same replacement reuses its color across rows and columns: a chip on
    an original value always matches the chip on the value that replaced it.
    """

    def __init__(self) -> None:
        self._assigned: dict[tuple[str, str], str] = {}

    def color_for(self, original: str, transformed: str) -> str:
        key = (original.strip().lower(), transformed.strip().lower())
        color = self._assigned.get(key)
        if color is None:
            color = _PAIR_BORDER_COLORS[len(self._assigned) % len(_PAIR_BORDER_COLORS)]
            self._assigned[key] = color
        return color


def _chip(value: str, border: str) -> str:
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


_needle_patterns: dict[str, re.Pattern[str]] = {}


def _needle_pattern(needle: str) -> re.Pattern[str]:
    """Compile a word-boundary matcher for ``needle``.

    Mirrors the word-boundary rule free-text propagation applies in
    ``replacement.apply``, so the preview only chips spans the replacer would
    actually have replaced. Plain substring matching highlighted misleading
    fragments, such as ``Lab`` inside ``Laboratory``.
    """
    pattern = _needle_patterns.get(needle)
    if pattern is None:
        pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", flags=re.IGNORECASE)
        _needle_patterns[needle] = pattern
    return pattern


def _highlight_with_pairs(
    text: str,
    pairs: list[tuple[str, str]],
    *,
    side: str,
    pair_colors: _PairColors,
) -> str:
    if not text or not pairs:
        return _plain(text)

    # Keep each needle tied to its pair so the chip color identifies the
    # replacement rather than the column it appears in.
    candidates = [
        (original if side == _SIDE_ORIGINAL else synthetic, (original, synthetic)) for original, synthetic in pairs
    ]
    candidates = [(needle, pair) for needle, pair in candidates if needle]
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    if not candidates:
        return _plain(text)

    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        match_at: int | None = None
        match_len = 0
        match_pair: tuple[str, str] | None = None
        for needle, pair in candidates:
            found = _needle_pattern(needle).search(text, cursor)
            if found is None:
                continue
            start, length = found.start(), found.end() - found.start()
            if match_at is None or start < match_at or (start == match_at and length > match_len):
                match_at = start
                match_len = length
                match_pair = pair
        if match_at is None or match_pair is None:
            parts.append(html.escape(text[cursor:]))
            break
        if match_at > cursor:
            parts.append(html.escape(text[cursor:match_at]))
        border = pair_colors.color_for(*match_pair)
        parts.append(_chip(text[match_at : match_at + match_len], border))
        cursor = match_at + match_len
    return "".join(parts) if parts else _plain(text)


def _render_cell_html(
    column: str,
    *,
    original_value: object,
    transformed_value: object,
    side: str,
    column_statistics: dict[str, ColumnStatistics],
    free_text_entities: list[dict[str, Any]],
    pair_colors: _PairColors,
) -> str:
    original_text = _cell_str(original_value)
    transformed_text = _cell_str(transformed_value)
    text = original_text if side == _SIDE_ORIGINAL else transformed_text
    changed = original_text != transformed_text
    stats = column_statistics.get(column)
    is_free_text = bool(stats and stats.assigned_entity == "free_text")

    # Free-text pairs are pooled per column across records, so only localize
    # them in cells that actually changed; an untouched cell gets no chips even
    # if it happens to contain a value replaced in some other record.
    if is_free_text and changed:
        pairs = _free_text_pairs_for_column(free_text_entities, column)
        if pairs:
            return _highlight_with_pairs(text, pairs, side=side, pair_colors=pair_colors)

    if changed:
        return _chip(text, pair_colors.color_for(original_text, transformed_text))
    return _plain(text)


def build_interlaced_comparison_frame(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a column-interlaced frame: an ``original`` / ``transformed`` pair per column.

    Rows are records (``original_df.index``). Columns are a ``MultiIndex`` of
    ``(column, side)`` where ``side`` is ``original`` or ``transformed``, so each
    source column appears as two adjacent sub-columns.
    """
    cols = columns or [c for c in original_df.columns if c in transformed_df.columns]
    if not cols:
        return pd.DataFrame(index=original_df.index)
    transformed_aligned = transformed_df.reindex(original_df.index)
    data: dict[tuple[str, str], pd.Series] = {}
    for col in cols:
        data[(col, _SIDE_ORIGINAL)] = original_df[col]
        data[(col, _SIDE_TRANSFORMED)] = transformed_aligned[col]
    frame = pd.DataFrame(data, index=original_df.index)
    frame.columns = pd.MultiIndex.from_tuples(list(data.keys()), names=["column", "side"])
    return frame


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
_ORIGINAL_BG = "#ffffff"
_TRANSFORMED_BG = "#f4faf9"
# Chip colors identify value pairs, so column pairs are grouped by a divider.
_PAIR_DIVIDER = "border-left:2px solid #c9d2dc;"
_ENTITY_LABEL_COLOR = "#5b6b7c"


def _column_width_px(column: str, column_statistics: dict[str, ColumnStatistics]) -> int:
    stats = column_statistics.get(column)
    if stats is not None and stats.assigned_entity == "free_text":
        return _TEXT_COL_PX
    return _DATA_COL_PX


def _render_group_header_cell(column: str, column_statistics: dict[str, ColumnStatistics]) -> str:
    """Top header cell spanning a column's original/transformed pair."""
    entity = _entity_for_column(column_statistics, column)
    entity_html = (
        f'<div style="margin-top:2px;font-size:11px;font-weight:600;color:{_ENTITY_LABEL_COLOR}">'
        f"{html.escape(entity)}</div>"
        if entity
        else ""
    )
    return (
        f'<th scope="colgroup" colspan="2" style="{_CELL_BASE}{_HEADER_BG}{_PAIR_DIVIDER}">'
        f'<div style="font-weight:700;color:#1f2a37">{html.escape(column)}</div>'
        f"{entity_html}"
        f"</th>"
    )


def _render_side_header_cell(side: str) -> str:
    bg = _TRANSFORMED_BG if side == _SIDE_TRANSFORMED else _ORIGINAL_BG
    color = "#0f6a6a" if side == _SIDE_TRANSFORMED else "#6b7c8a"
    divider = _PAIR_DIVIDER if side == _SIDE_ORIGINAL else ""
    return (
        f'<th scope="col" style="{_CELL_BASE}background:{bg};{divider}">'
        f'<div style="font-size:10.5px;text-transform:uppercase;letter-spacing:0.04em;'
        f'font-weight:600;color:{color}">{html.escape(side)}</div>'
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
    """Render one column-interlaced Original/Transformed HTML table with chips.

    Each source column is shown as an ``original`` / ``transformed`` sub-column
    pair; one row per record. Chip colors are assigned per distinct value pair,
    so a replaced value and its replacement share a color. All styling is
    inline: notebook HTML sanitizers commonly strip ``<style>`` blocks, which
    previously dropped the header styling and column widths.
    """
    free_text_entities = free_text_entities or []
    cols = columns or [c for c in original_df.columns if c in transformed_df.columns]
    transformed_aligned = transformed_df.reindex(original_df.index)

    # colgroup: record column + two equal-width sub-columns per source column.
    col_specs = [f'<col style="width:{_INDEX_COL_PX}px" />']
    for col in cols:
        width = _column_width_px(col, column_statistics)
        col_specs.append(f'<col style="width:{width}px" />')
        col_specs.append(f'<col style="width:{width}px" />')
    colgroup = "<colgroup>" + "".join(col_specs) + "</colgroup>"
    total_px = _INDEX_COL_PX + sum(2 * _column_width_px(col, column_statistics) for col in cols)

    pair_colors = _PairColors()

    group_header = [f'<th scope="col" rowspan="2" style="{_INDEX_CELL_BASE}{_HEADER_BG}">record</th>'] + [
        _render_group_header_cell(col, column_statistics) for col in cols
    ]
    side_header: list[str] = []
    for _ in cols:
        side_header.append(_render_side_header_cell(_SIDE_ORIGINAL))
        side_header.append(_render_side_header_cell(_SIDE_TRANSFORMED))

    body_rows: list[str] = []
    for record_id in original_df.index:
        original_row = original_df.loc[record_id]
        transformed_row = transformed_aligned.loc[record_id]
        cells: list[str] = [
            f'<th scope="row" style="{_INDEX_CELL_BASE}background:#fafbfc;">'
            f'<div style="font-weight:700;color:#1f2a37">{html.escape(str(record_id))}</div>'
            f"</th>"
        ]
        for col in cols:
            original_value = original_row.get(col)
            transformed_value = transformed_row.get(col)
            cells.append(
                f'<td style="{_CELL_BASE}background:{_ORIGINAL_BG};{_PAIR_DIVIDER}">'
                + _render_cell_html(
                    col,
                    original_value=original_value,
                    transformed_value=transformed_value,
                    side=_SIDE_ORIGINAL,
                    column_statistics=column_statistics,
                    free_text_entities=free_text_entities,
                    pair_colors=pair_colors,
                )
                + "</td>"
            )
            cells.append(
                f'<td style="{_CELL_BASE}background:{_TRANSFORMED_BG};">'
                + _render_cell_html(
                    col,
                    original_value=original_value,
                    transformed_value=transformed_value,
                    side=_SIDE_TRANSFORMED,
                    column_statistics=column_statistics,
                    free_text_entities=free_text_entities,
                    pair_colors=pair_colors,
                )
                + "</td>"
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
<div style="font-family:'IBM Plex Sans','Segoe UI',sans-serif;color:#1f2a37;border:1px solid #c9d2dc;\
background:#f7f9fb;border-radius:8px;padding:12px;max-width:100%">
  <div style="font-size:13px;color:#5b6b7c;margin-bottom:10px">Showing {len(original_df)} record(s) \
&times; {len(cols)} column(s); each column shown as original / transformed. Matching chip colors link a \
replaced value to its replacement.</div>
  <div style="border:1px solid #c9d2dc;border-radius:8px;overflow:hidden;background:#fff">
    <div style="padding:10px 14px;border-bottom:1px solid #c9d2dc;font-weight:650;background:#f3f6f9">\
{html.escape(title)}</div>
    <div style="padding:12px 14px">
      <div style="overflow:auto;max-height:34rem;border:1px solid #c9d2dc;border-radius:6px">
        <table style="border-collapse:collapse;table-layout:fixed;width:{total_px}px;\
font-family:'IBM Plex Mono','SFMono-Regular',Menlo,Consolas,monospace;font-size:12.5px;line-height:1.45">
          {colgroup}
          <thead>
            <tr>{"".join(group_header)}</tr>
            <tr>{"".join(side_header)}</tr>
          </thead>
          <tbody>{"".join(body_rows)}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>
""".strip()


@dataclass
class PiiReplacementResultPreview:
    """Notebook display object for the column-interlaced original/transformed table."""

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
    """Show original vs transformed columns after PII replacement (``.head()``-style)."""
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

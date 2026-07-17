# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook and HTML rendering for resolved tabular PII replacement plans."""

from __future__ import annotations

import html
import importlib
import sys
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config.data import DataParameters
from ..config.pii_replacement import (
    PiiColumnPlan,
    PiiEntity,
    PiiReplacementPlan,
    ReplacePiiConfig,
    is_person_entity,
)
from . import core
from .plan import resolve_plan, unique_id_advisories
from .runtime_config import runtime_config_from_replace_pii

_PALETTE = (
    ("#dbeafe", "#2563eb"),
    ("#dcfce7", "#16a34a"),
    ("#fef3c7", "#d97706"),
    ("#fce7f3", "#db2777"),
    ("#e0e7ff", "#4f46e5"),
    ("#d1fae5", "#059669"),
    ("#ffedd5", "#ea580c"),
    ("#ede9fe", "#7c3aed"),
)
_STANDALONE_COLORS = ("#f3f4f6", "#6b7280")
_GROUP_KEY_COLORS = ("#cffafe", "#0891b2")
_UNAFFECTED_COLORS = ("#f9fafb", "#9ca3af")


@dataclass(frozen=True)
class _ColumnGroup:
    label: str
    columns: tuple[str, ...]
    background: str
    border: str
    kind: str


@dataclass
class PiiPlanPreview:
    """Resolved replacement plan plus original sample rows for safe inspection.

    The preview never applies replacements. In notebooks, evaluating this object
    renders its HTML representation; [`display`][PiiPlanPreview.display] provides
    an explicit equivalent.
    """

    plan: PiiReplacementPlan
    sample_dataframe: pd.DataFrame
    total_rows: int
    advisories: list[str] = field(default_factory=list)
    include_unaffected: bool = True
    column_stats: dict[str, dict[str, object]] = field(default_factory=dict)
    dominant_pattern_min_coverage: float = 85.0

    def to_html(self) -> str:
        """Return a self-contained HTML view of the plan and original sample rows."""
        return render_plan_preview_html(
            self.plan,
            self.sample_dataframe,
            total_rows=self.total_rows,
            advisories=self.advisories,
            include_unaffected=self.include_unaffected,
            column_stats=self.column_stats,
            dominant_pattern_min_coverage=self.dominant_pattern_min_coverage,
        )

    def _repr_html_(self) -> str:
        """Render the preview when it is the final expression in a notebook cell."""
        return self.to_html()

    def to_text(self) -> str:
        """Return a compact plain-text representation for non-notebook environments."""
        whole_columns = _whole_column_count(self.plan, min_coverage=self.dominant_pattern_min_coverage)
        mixed_columns = _mixed_column_count(self.plan, min_coverage=self.dominant_pattern_min_coverage)
        free_text_columns = _free_text_count(self.plan)
        lines = [
            "PII Replacement Plan Preview",
            f"Rows: {self.total_rows}",
            f"Group key: {self.plan.group_key or 'None'}",
            f"Entire-column replacements: {whole_columns}",
            f"Mixed-value columns: {mixed_columns}",
            f"Free-text propagation columns: {free_text_columns}",
        ]
        for column, spec in self.plan.columns.items():
            scope = _scope_label(spec, min_coverage=self.dominant_pattern_min_coverage)
            entity = spec.entity_type.value if spec.entity_type else "unclassified"
            relation = spec.persona or "independent"
            column_scope = self.column_stats.get(column, {}).get("scope", "unknown")
            lines.append(f"- {column}: {entity}; {scope}; {relation}; scope={column_scope}")
        lines.extend(f"! {message}" for message in self.advisories)
        return "\n".join(lines)

    def display(self) -> None:
        """Display rich HTML in IPython, falling back to a plain-text plan."""
        try:
            ipython_display = importlib.import_module("IPython.display")
        except ImportError:
            sys.stdout.write(self.to_text() + "\n")
            return
        ipython_display.display(ipython_display.HTML(self.to_html()))


def build_plan_preview(
    df: pd.DataFrame,
    config: ReplacePiiConfig,
    *,
    data_config: DataParameters,
    num_rows: int = 5,
    include_unaffected: bool = True,
) -> PiiPlanPreview:
    """Resolve a replacement plan and build a notebook-renderable preview.

    This is the plan-only entry point: it never calls ``run_replacement()``.

    Args:
        df: Input dataframe used for plan discovery and sample rows.
        config: Top-level ``replace_pii`` configuration.
        data_config: Data grouping configuration used to backfill ``group_key``.
        num_rows: Maximum number of original rows to include in the preview.
        include_unaffected: Whether the sample table includes columns outside
            the resolved replacement plan.

    Returns:
        A resolved-plan preview with original sample values only.

    Raises:
        ValueError: If ``num_rows`` is negative.
    """
    if num_rows < 0:
        raise ValueError("num_rows must be non-negative")

    runtime = runtime_config_from_replace_pii(config)
    plan = resolve_plan(config, df, data_config=data_config, runtime=runtime)
    return PiiPlanPreview(
        plan=plan,
        sample_dataframe=df.head(num_rows).copy(),
        total_rows=len(df),
        advisories=unique_id_advisories(df, plan, runtime),
        include_unaffected=include_unaffected,
        column_stats=core.scoped_column_stats(df, plan.group_key),
        dominant_pattern_min_coverage=runtime.dominant_pattern_min_coverage,
    )


def render_plan_preview_html(
    plan: PiiReplacementPlan,
    sample_dataframe: pd.DataFrame,
    *,
    total_rows: int | None = None,
    advisories: list[str] | None = None,
    include_unaffected: bool = True,
    column_stats: dict[str, dict[str, object]] | None = None,
    dominant_pattern_min_coverage: float = 85.0,
) -> str:
    """Render a resolved plan without computing or exposing replacement values."""
    stats = column_stats or {}
    groups = _sample_groups(sample_dataframe, plan, include_unaffected=include_unaffected)
    row_count = len(sample_dataframe) if total_rows is None else total_rows
    summary = _render_summary(
        plan,
        row_count=row_count,
        sample_rows=len(sample_dataframe),
        min_coverage=dominant_pattern_min_coverage,
    )
    advisory_html = _render_advisories(advisories or [])
    persona_html = _render_persona_groups(plan)
    manifest_html = _render_manifest(
        plan,
        sample_dataframe,
        column_stats=stats,
        min_coverage=dominant_pattern_min_coverage,
    )
    sample_html = _render_sample_table(
        sample_dataframe,
        plan,
        groups,
        min_coverage=dominant_pattern_min_coverage,
    )

    return f"""\
<style>
  .nss-pii-plan-preview {{
    --nss-fg: var(--jp-ui-font-color1, var(--vscode-editor-foreground, #1e293b));
    --nss-muted: var(--jp-ui-font-color2, var(--vscode-descriptionForeground, #64748b));
    --nss-surface: var(--jp-layout-color1, var(--vscode-editor-background, #ffffff));
    --nss-subtle-surface: var(--jp-layout-color2, var(--vscode-sideBar-background, #f8fafc));
    --nss-border: var(--jp-border-color2, var(--vscode-panel-border, #d1d5db));
  }}
  @media (prefers-color-scheme: dark) {{
    .nss-pii-plan-preview {{
      --nss-fg: var(--jp-ui-font-color1, var(--vscode-editor-foreground, #e2e8f0));
      --nss-muted: var(--jp-ui-font-color2, var(--vscode-descriptionForeground, #94a3b8));
      --nss-surface: var(--jp-layout-color1, var(--vscode-editor-background, #0f172a));
      --nss-subtle-surface: var(--jp-layout-color2, var(--vscode-sideBar-background, #1e293b));
      --nss-border: var(--jp-border-color2, var(--vscode-panel-border, #475569));
    }}
  }}
</style>
<div class="nss-pii-plan-preview" style="font-family:system-ui,-apple-system,sans-serif;\
max-width:1200px;margin:12px 0;color:var(--nss-fg);background:var(--nss-surface)">
  <div style="border:1px solid var(--nss-border);border-radius:10px;overflow:hidden">
    <div style="padding:12px 18px;border-bottom:1px solid var(--nss-border);\
background:var(--nss-subtle-surface)">
      <strong style="font-size:1.05em">PII Replacement Plan Preview</strong>
      <div style="font-size:0.82em;color:var(--nss-muted);margin-top:3px">Detection and plan only — no replacements have been applied.</div>
    </div>
    <div style="padding:16px 18px">
      {summary}
      {advisory_html}
      {persona_html}
      {_section("Column Plan", manifest_html)}
      {_section("Original Data Sample", sample_html)}
    </div>
  </div>
</div>"""


def _section(title: str, body: str) -> str:
    return (
        "<div style='margin-top:18px'>"
        "<div style='font-size:0.78em;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.06em;color:var(--nss-muted);margin-bottom:7px'>{html.escape(title)}</div>"
        f"{body}</div>"
    )


def _whole_column_count(plan: PiiReplacementPlan, *, min_coverage: float) -> int:
    return sum(_scope_label(spec, min_coverage=min_coverage) == "Entire column" for spec in plan.columns.values())


def _mixed_column_count(plan: PiiReplacementPlan, *, min_coverage: float) -> int:
    return sum(_scope_label(spec, min_coverage=min_coverage) == "Mixed values" for spec in plan.columns.values())


def _free_text_count(plan: PiiReplacementPlan) -> int:
    return sum(spec.entity_type == PiiEntity.free_text for spec in plan.columns.values())


def _render_summary(plan: PiiReplacementPlan, *, row_count: int, sample_rows: int, min_coverage: float) -> str:
    group_key = html.escape(plan.group_key) if plan.group_key else "None"
    values = (
        ("Rows", str(row_count)),
        ("Sample rows", str(sample_rows)),
        ("Group key", group_key),
        ("Personas", str(len(plan.identified_personas))),
        ("Entire columns", str(_whole_column_count(plan, min_coverage=min_coverage))),
        ("Mixed columns", str(_mixed_column_count(plan, min_coverage=min_coverage))),
        ("Free-text columns", str(_free_text_count(plan))),
    )
    cards = "".join(
        "<div style='border:1px solid var(--nss-border);border-radius:7px;padding:8px 11px;min-width:100px'>"
        f"<div style='font-size:0.72em;text-transform:uppercase;color:var(--nss-muted)'>{label}</div>"
        f"<div style='font-weight:650;margin-top:2px'>{value}</div></div>"
        for label, value in values
    )
    return f"<div style='display:flex;gap:8px;flex-wrap:wrap'>{cards}</div>"


def _render_advisories(advisories: list[str]) -> str:
    if not advisories:
        return ""
    items = "".join(f"<li>{html.escape(message)}</li>" for message in advisories)
    return (
        "<div style='margin-top:14px;padding:9px 12px;border-left:4px solid #d97706;"
        f"background:#fffbeb;color:#1e293b;border-radius:4px;font-size:0.86em'>"
        f"<strong>Advisories</strong><ul>{items}</ul></div>"
    )


def _persona_names(plan: PiiReplacementPlan) -> list[str]:
    names = list(plan.identified_personas)
    for spec in plan.columns.values():
        if spec.persona and spec.persona not in names:
            names.append(spec.persona)
    return names


def _persona_colors(plan: PiiReplacementPlan) -> dict[str, tuple[str, str]]:
    return {name: _PALETTE[index % len(_PALETTE)] for index, name in enumerate(_persona_names(plan))}


def _conditioning_columns(plan: PiiReplacementPlan, persona_name: str) -> list[tuple[str, str]]:
    persona = plan.identified_personas.get(persona_name)
    if persona is None:
        return []
    result: list[tuple[str, str]] = []
    if persona.gender:
        result.append(("Gender", persona.gender))
    if persona.ethnic_background:
        result.append(("Ethnic background", persona.ethnic_background))
    return result


def _render_persona_groups(plan: PiiReplacementPlan) -> str:
    names = _persona_names(plan)
    if not names:
        return ""
    colors = _persona_colors(plan)
    cards: list[str] = []
    for name in names:
        background, border = colors[name]
        replacement_columns = [column for column, spec in plan.columns.items() if spec.persona == name]
        conditioning = _conditioning_columns(plan, name)
        replacement_text = ", ".join(html.escape(column) for column in replacement_columns) or "No replacement columns"
        conditioning_text = (
            ", ".join(f"{html.escape(label)}: {html.escape(column)}" for label, column in conditioning)
            or "No conditioning columns"
        )
        cards.append(
            f"<div style='border:1px solid {border};border-left-width:5px;border-radius:7px;"
            f"padding:9px 12px;background:{background};color:#1e293b;min-width:230px'>"
            f"<div style='font-weight:700'>{html.escape(name)}</div>"
            f"<div style='font-size:0.8em;margin-top:4px'><strong>Related columns:</strong> {replacement_text}</div>"
            f"<div style='font-size:0.8em;margin-top:2px'><strong>Conditions:</strong> {conditioning_text}</div>"
            "</div>"
        )
    return _section(
        "Related Persona Groups", f"<div style='display:flex;gap:8px;flex-wrap:wrap'>{''.join(cards)}</div>"
    )


def _scope_label(spec: PiiColumnPlan, *, min_coverage: float = 85.0) -> str:
    if spec.entity_type is None:
        return "No action"
    if spec.entity_type == PiiEntity.free_text:
        return "Matching values only"
    if spec.entity_type == PiiEntity.date:
        return "Identified only"
    if is_person_entity(spec.entity_type):
        return "Entire column"
    if spec.dominant_pattern_coverage is not None and spec.dominant_pattern_coverage < min_coverage:
        return "Mixed values"
    return "Entire column"


def _scope_tag(spec: PiiColumnPlan | None, *, min_coverage: float) -> str:
    if spec is None or spec.entity_type is None:
        return ""
    label = _scope_label(spec, min_coverage=min_coverage)
    if label == "Entire column":
        return "FULL"
    if label == "Mixed values":
        return "MIXED"
    if label == "Matching values only":
        return "MATCHES"
    if label == "Identified only":
        return "IDENTIFY"
    return label.upper()


def _method_label(spec: PiiColumnPlan) -> str:
    if spec.entity_type is None:
        return "None"
    if spec.entity_type == PiiEntity.free_text:
        return "Propagation"
    if spec.entity_type == PiiEntity.date:
        return "None"
    if spec.entity_type == PiiEntity.date_of_birth:
        return "Age-preserving perturbation"
    if is_person_entity(spec.entity_type):
        return "Persona synthesis"
    return "Pattern synthesis" if spec.pattern else "Synthetic replacement"


def _badge(text: str, *, background: str, border: str) -> str:
    return (
        f"<span style='display:inline-block;padding:2px 7px;border:1px solid {border};"
        f"border-radius:999px;background:{background};color:#1e293b;font-size:0.78em;font-weight:650'>"
        f"{html.escape(text)}</span>"
    )


def _sample_values_text(column: str, column_stats: dict[str, dict[str, object]]) -> str:
    raw = column_stats.get(column, {}).get("samples", [])
    if not isinstance(raw, list) or not raw:
        return "—"
    values = [html.escape(str(value)) for value in raw[:4]]
    if len(raw) > 4:
        values.append("…")
    return ", ".join(values)


def _column_scope_text(column: str, column_stats: dict[str, dict[str, object]]) -> str:
    scope = column_stats.get(column, {}).get("scope")
    if not scope:
        return "—"
    return html.escape(str(scope))


def _render_manifest(
    plan: PiiReplacementPlan,
    dataframe: pd.DataFrame,
    *,
    column_stats: dict[str, dict[str, object]],
    min_coverage: float,
) -> str:
    if not plan.columns:
        return (
            "<p style='color:var(--nss-muted);font-style:italic'>No columns are included in the replacement plan.</p>"
        )
    colors = _persona_colors(plan)
    ordered_columns = [column for column in dataframe.columns if column in plan.columns]
    ordered_columns.extend(column for column in plan.columns if column not in ordered_columns)
    rows: list[str] = []
    for column in ordered_columns:
        spec = plan.columns[column]
        entity = spec.entity_type.value if spec.entity_type else "unclassified"
        persona = spec.persona or "Independent"
        background, border = colors.get(spec.persona or "", _STANDALONE_COLORS)
        relation = _badge(persona, background=background, border=border)
        if column == plan.group_key:
            relation += " " + _badge("Group key", background=_GROUP_KEY_COLORS[0], border=_GROUP_KEY_COLORS[1])
        scope_label = _scope_label(spec, min_coverage=min_coverage)
        if scope_label == "Entire column":
            scope_background, scope_border = "#fee2e2", "#dc2626"
        elif scope_label == "Mixed values":
            scope_background, scope_border = "#ffedd5", "#ea580c"
        else:
            scope_background, scope_border = "#fef3c7", "#d97706"
        pattern = html.escape(spec.pattern) if spec.pattern else "—"
        if spec.dominant_pattern_coverage is not None:
            pattern += f" ({spec.dominant_pattern_coverage:.1f}%)"
        rows.append(
            f"<tr style='border-top:1px solid #e5e7eb;border-left:4px solid {border}'>"
            f"<td style='padding:7px 9px;font-family:ui-monospace,monospace;font-weight:650'>{html.escape(column)}</td>"
            f"<td style='padding:7px 9px'>{_badge(entity, background=background, border=border)}</td>"
            f"<td style='padding:7px 9px'>{_badge(scope_label, background=scope_background, border=scope_border)}</td>"
            f"<td style='padding:7px 9px'>{html.escape(_method_label(spec))}</td>"
            f"<td style='padding:7px 9px'>{relation}</td>"
            f"<td style='padding:7px 9px;font-size:0.84em'>{_column_scope_text(column, column_stats)}</td>"
            f"<td style='padding:7px 9px;font-family:ui-monospace,monospace;font-size:0.84em'>{pattern}</td>"
            f"<td style='padding:7px 9px;font-family:ui-monospace,monospace;font-size:0.84em'>"
            f"{_sample_values_text(column, column_stats)}</td>"
            "</tr>"
        )
    return (
        "<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;font-size:0.86em'>"
        "<thead><tr style='text-align:left;background:var(--nss-subtle-surface);color:var(--nss-fg)'>"
        "<th style='padding:7px 9px'>Column</th><th style='padding:7px 9px'>Entity type</th>"
        "<th style='padding:7px 9px'>Replacement scope</th><th style='padding:7px 9px'>Method</th>"
        "<th style='padding:7px 9px'>Related group</th><th style='padding:7px 9px'>Column scope</th>"
        "<th style='padding:7px 9px'>Pattern</th><th style='padding:7px 9px'>Sample values</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _sample_groups(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    *,
    include_unaffected: bool,
) -> list[_ColumnGroup]:
    available = set(dataframe.columns)
    used: set[str] = set()
    groups: list[_ColumnGroup] = []

    if plan.group_key and plan.group_key in available:
        groups.append(_ColumnGroup("Dataset group key", (plan.group_key,), *_GROUP_KEY_COLORS, kind="group_key"))
        used.add(plan.group_key)

    colors = _persona_colors(plan)
    for persona_name in _persona_names(plan):
        related = [column for column, spec in plan.columns.items() if spec.persona == persona_name]
        related.extend(column for _, column in _conditioning_columns(plan, persona_name))
        columns = tuple(column for column in related if column in available and column not in used)
        if not columns:
            continue
        groups.append(_ColumnGroup(persona_name, columns, *colors[persona_name], kind="persona"))
        used.update(columns)

    standalone = tuple(column for column in plan.columns if column in available and column not in used)
    if standalone:
        groups.append(_ColumnGroup("Independent PII", standalone, *_STANDALONE_COLORS, kind="standalone"))
        used.update(standalone)

    if include_unaffected:
        unaffected = tuple(column for column in dataframe.columns if column not in used)
        if unaffected:
            groups.append(_ColumnGroup("Unaffected context", unaffected, *_UNAFFECTED_COLORS, kind="unaffected"))

    return groups


def _format_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "<span style='color:var(--nss-muted)'>—</span>"
    except (TypeError, ValueError):
        pass
    text = str(value)
    if len(text) > 100:
        text = text[:97] + "..."
    return html.escape(text)


def _render_sample_table(
    dataframe: pd.DataFrame,
    plan: PiiReplacementPlan,
    groups: list[_ColumnGroup],
    *,
    min_coverage: float,
) -> str:
    if not groups:
        return "<p style='color:var(--nss-muted);font-style:italic'>No columns are available to preview.</p>"

    group_headers = "".join(
        f"<th colspan='{len(group.columns)}' style='padding:6px 8px;text-align:center;"
        f"background:{group.background};border-bottom:3px solid {group.border};"
        f"color:#1e293b'>{html.escape(group.label)}</th>"
        for group in groups
    )
    column_headers: list[str] = []
    ordered_columns: list[str] = []
    column_group: dict[str, _ColumnGroup] = {}
    for group in groups:
        for column in group.columns:
            ordered_columns.append(column)
            column_group[column] = group
            spec = plan.columns.get(column)
            tags: list[str] = []
            if spec and spec.entity_type:
                tags.append(spec.entity_type.value)
                scope_tag = _scope_tag(spec, min_coverage=min_coverage)
                if scope_tag:
                    tags.append(scope_tag)
            elif group.kind == "persona":
                tags.append("CONDITION")
            if column == plan.group_key:
                tags.append("GROUP KEY")
            tag_html = "".join(
                f"<div style='font-size:0.68em;color:{group.border};margin-top:2px'>{html.escape(tag)}</div>"
                for tag in tags
            )
            column_headers.append(
                f"<th style='padding:7px 9px;text-align:left;vertical-align:top;white-space:nowrap;"
                f"background:{group.background};color:#1e293b;border-bottom:1px solid {group.border}'>"
                f"<span style='font-family:ui-monospace,monospace'>{html.escape(column)}</span>{tag_html}</th>"
            )

    rows: list[str] = []
    for _, row in dataframe.loc[:, ordered_columns].iterrows():
        cells = "".join(
            f"<td style='padding:7px 9px;border-top:1px solid #e5e7eb;"
            f"border-left:2px solid {column_group[column].background};"
            f"max-width:280px;white-space:pre-wrap;overflow-wrap:anywhere'>{_format_cell(row[column])}</td>"
            for column in ordered_columns
        )
        rows.append(f"<tr>{cells}</tr>")

    if not rows:
        rows.append(
            f"<tr><td colspan='{len(ordered_columns)}' style='padding:12px;color:var(--nss-muted);"
            "font-style:italic;text-align:center'>No sample rows.</td></tr>"
        )

    return (
        "<div style='overflow-x:auto'><table style='border-collapse:separate;border-spacing:0;"
        "width:max-content;min-width:100%;font-size:0.83em'>"
        f"<thead><tr>{group_headers}</tr><tr>{''.join(column_headers)}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<div style='font-size:0.75em;color:var(--nss-muted);margin-top:6px'>"
        "Values shown are from the original data. FULL marks entire-column replacement; "
        "MIXED marks per-value replacement; MATCHES marks in-cell propagation.</div>"
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive notebook preview for PII replacement plans in a Safe Synthesizer run.

The preview shows editable YAML beside a UML-style card diagram of
``persona_backed_columns`` and ``standalone_columns_to_replace``. Each card and
the scope chip carry a ``?`` toggle holding that section's explanation, so the
diagram can explain the plan rather than only echoing field names. Validation
always runs against the attached training dataframe (schema +
``validate_plan``). Section placement mismatches surface as non-blocking
warnings on the widget (same rules as preflight / apply logging).
Hovering a diagram region highlights the matching YAML span (and vice versa).
Edits are applied when the user clicks **Save and render diagram**; each valid
render can sync the plan back into the Safe Synthesizer builder.

Prefer :meth:`SafeSynthesizer.preview_replace_pii` so discovery, data config, and
plan sync stay inside the builder workflow. Preview is optional — skip it and
``run()`` / ``process_data()`` still use auto-discovery or the configured plan.

Install the optional notebook extra before use::

    pip install 'nemo-safe-synthesizer[notebook]'
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import ValidationError

from ..config.data import DataParameters
from ..config.pii_replacement import PersonaColumnSet, PiiColumnPlan, PiiReplacementPlan
from ..config.time_series import TimeSeriesParameters
from ..errors import ParameterError
from ..pii_replacer.planning import iter_plan_advisories, plan_section_help, validate_plan

__all__ = [
    "PreviewState",
    "PiiPlanPreview",
    "build_diagram_model",
    "map_yaml_source_ranges",
    "plan_to_preview_yaml",
    "preview_pii_replacement_plan",
    "preview_state_from_yaml",
]

_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "pii_plan_preview"
_NOTEBOOK_EXTRA_HINT = (
    "Notebook PII plan preview requires the 'notebook' optional dependency. "
    "Install with: pip install 'nemo-safe-synthesizer[notebook]'"
)

PlanCallback = Callable[[PiiReplacementPlan], None]


@dataclass(frozen=True)
class PreviewState:
    """Parsed preview payload synced to the notebook widget."""

    yaml_text: str
    diagram: dict[str, Any]
    ranges: dict[str, dict[str, int]]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    plan: PiiReplacementPlan | None = None
    valid: bool = False


def _status_for_state(state: PreviewState, *, initial: bool = False) -> str:
    if not state.valid:
        return "Fix validation errors, then click Save and render diagram."
    if state.warnings:
        n = len(state.warnings)
        label = "warning" if n == 1 else "warnings"
        if initial:
            return (
                f"Plan is valid with {n} placement {label}. "
                "Edit and click Save and render diagram to update, then continue the Safe Synthesizer run."
            )
        return (
            f"Plan is valid with {n} placement {label}. "
            "Continue the Safe Synthesizer run, or edit and Save and render diagram again."
        )
    if initial:
        return (
            "Plan is valid. Edit and click Save and render diagram to update, then continue the Safe Synthesizer run."
        )
    return "Plan is valid. Continue the Safe Synthesizer run, or edit and Save and render diagram again."


def plan_to_preview_yaml(plan: PiiReplacementPlan) -> str:
    """Serialize a plan the same way artifact writers omit defaults/nulls.

    Section comments are left out here; the diagram's ``?`` toggles carry those
    explanations instead.
    """
    data = json.loads(plan.model_dump_json(exclude_none=True, exclude_defaults=True))
    return yaml.safe_dump(data, sort_keys=False)


def map_yaml_source_ranges(text: str) -> dict[str, dict[str, int]]:
    """Map canonical plan paths to inclusive-start / exclusive-end character offsets.

    Paths use JSON-pointer-like indexing, e.g.
    ``persona_backed_columns[0].columns_to_replace[1].column_name``.
    """
    if not text.strip():
        return {}
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return {}
    if root is None:
        return {}

    ranges: dict[str, dict[str, int]] = {}

    def _add(path: str, start: int | None, end: int | None) -> None:
        if not path or start is None or end is None:
            return
        start_i, end_i = int(start), int(end)
        if end_i < start_i:
            return
        existing = ranges.get(path)
        # Prefer the wider span so mapping entries keep ``key: value`` rather than
        # being overwritten by the scalar value node alone.
        if existing is not None and existing["end"] - existing["start"] >= end_i - start_i:
            return
        ranges[path] = {"start": start_i, "end": end_i}

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, yaml.MappingNode):
            if path:
                _add(path, node.start_mark.index, node.end_mark.index)
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    continue
                key = str(key_node.value)
                child = f"{path}.{key}" if path else key
                _add(child, key_node.start_mark.index, value_node.end_mark.index)
                _walk(value_node, child)
        elif isinstance(node, yaml.SequenceNode):
            if path:
                _add(path, node.start_mark.index, node.end_mark.index)
            for index, item in enumerate(node.value):
                child = f"{path}[{index}]"
                _add(child, item.start_mark.index, item.end_mark.index)
                _walk(item, child)
        elif isinstance(node, yaml.ScalarNode):
            if path:
                _add(path, node.start_mark.index, node.end_mark.index)

    _walk(root, "")
    return ranges


def _column_row(path: str, spec: PiiColumnPlan) -> dict[str, Any]:
    secondary = spec.entity_type.value if spec.entity_type is not None else None
    return {
        "path": path,
        "primary": spec.column_name,
        "secondary": secondary,
    }


def _persona_card(index: int, col_set: PersonaColumnSet) -> dict[str, Any]:
    base = f"persona_backed_columns[{index}]"
    replace_rows = [
        _column_row(f"{base}.columns_to_replace[{j}]", spec) for j, spec in enumerate(col_set.columns_to_replace)
    ]
    match_rows = [
        {
            "path": f"{base}.match_persona_by[{j}]",
            "primary": entry.column_name,
            "secondary": entry.persona_attribute,
        }
        for j, entry in enumerate(col_set.match_persona_by)
    ]
    return {
        "kind": "persona",
        "path": base,
        "title": col_set.persona,
        "help": plan_section_help("persona_backed_columns"),
        "compartments": [
            {
                "path": f"{base}.columns_to_replace",
                "label": "Columns to replace",
                "rows": replace_rows,
            },
            {
                "path": f"{base}.match_persona_by",
                "label": "Match persona by",
                "hint": "used to pick the person, never replaced",
                "rows": match_rows,
            },
        ],
    }


def build_diagram_model(plan: PiiReplacementPlan) -> dict[str, Any]:
    """Build a presentation-only card model for the notebook diagram pane."""
    cards = [_persona_card(i, col_set) for i, col_set in enumerate(plan.persona_backed_columns)]
    standalone_rows = [
        _column_row(f"standalone_columns_to_replace[{j}]", spec)
        for j, spec in enumerate(plan.standalone_columns_to_replace)
    ]
    cards.append(
        {
            "kind": "standalone",
            "path": "standalone_columns_to_replace",
            "title": "Standalone columns",
            "help": plan_section_help("standalone_columns_to_replace"),
            "compartments": [
                {
                    "path": "standalone_columns_to_replace",
                    "label": "Columns to replace",
                    "rows": standalone_rows,
                }
            ],
        }
    )
    return {
        "scope": plan.scope.value,
        "scope_path": "scope",
        "scope_help": plan_section_help("scope"),
        "cards": cards,
    }


def _format_plan_parse_error(exc: BaseException) -> str:
    """Turn raw YAML/Pydantic parse failures into a preview-friendly message."""
    if isinstance(exc, yaml.YAMLError):
        detail = str(exc).strip()
        # Scanner errors like "mapping values are not allowed here" almost always
        # mean a key is indented past its siblings under a list item (e.g.
        # ``entity_type`` nested under ``column_name`` instead of beside it).
        hint = (
            "Invalid YAML near the marked line. Under a list item, keys "
            "`column_name`, `entity_type` and `patterns` must share the same "
            "indentation (two spaces past the `-`), not nest under each other."
        )
        return f"{hint}\n\n{detail}"
    return str(exc)


def preview_state_from_yaml(
    yaml_text: str,
    *,
    df: pd.DataFrame,
    data_config: DataParameters,
    previous: PreviewState | None = None,
    persona_backend: str = "managed",
    time_series: TimeSeriesParameters | None = None,
) -> PreviewState:
    """Parse YAML, validate against ``df``, and build diagram + source ranges.

    Invalid edits retain the last valid diagram and surface the error inline.
    Section placement mismatches become non-blocking ``warnings``.
    """
    ranges = map_yaml_source_ranges(yaml_text)

    def _invalid(exc: BaseException, plan: PiiReplacementPlan | None = None) -> PreviewState:
        # ``previous.diagram`` is the last diagram shown, which for an already
        # invalid previous state is still the last valid one. Reusing it keeps the
        # diagram stable across a run of failed edits instead of blanking it.
        message = _format_plan_parse_error(exc)
        if previous is not None:
            return PreviewState(
                yaml_text=yaml_text,
                diagram=previous.diagram,
                ranges=ranges or previous.ranges,
                error=message,
                warnings=[],
                plan=previous.plan,
                valid=False,
            )
        empty = PiiReplacementPlan()
        return PreviewState(
            yaml_text=yaml_text,
            diagram=build_diagram_model(empty),
            ranges=ranges,
            error=message,
            warnings=[],
            plan=plan,
            valid=False,
        )

    try:
        plan = PiiReplacementPlan.from_yaml_str(yaml_text)
    except (yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        return _invalid(exc)

    try:
        validate_plan(df, plan, data_config=data_config, time_series=time_series)
    except ParameterError as exc:
        return _invalid(exc, plan=plan)

    warnings = [issue.message for issue in iter_plan_advisories(plan, persona_backend=persona_backend)]
    return PreviewState(
        yaml_text=yaml_text,
        diagram=build_diagram_model(plan),
        ranges=ranges,
        error=None,
        warnings=warnings,
        plan=plan,
        valid=True,
    )


def _normalize_source(source: str | Path | PiiReplacementPlan) -> str:
    if isinstance(source, PiiReplacementPlan):
        return plan_to_preview_yaml(source)
    if isinstance(source, Path):
        return source.expanduser().read_text(encoding="utf-8")
    if isinstance(source, str):
        path = Path(source).expanduser()
        if "\n" not in source and path.is_file():
            return path.read_text(encoding="utf-8")
        return source
    raise TypeError(f"Unsupported preview source type: {type(source)!r}")


def _load_asset(name: str) -> str:
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


def _build_widget_class() -> type:
    try:
        import anywidget
        import traitlets
    except ImportError as exc:  # pragma: no cover - exercised via helper
        raise ImportError(_NOTEBOOK_EXTRA_HINT) from exc

    class _PiiPlanPreview(anywidget.AnyWidget):
        """Side-by-side YAML editor and interactive PII plan diagram."""

        _esm = _load_asset("widget.js")
        _css = _load_asset("widget.css")

        yaml_text = traitlets.Unicode("").tag(sync=True)
        diagram = traitlets.Dict(default_value={}).tag(sync=True)
        ranges = traitlets.Dict(default_value={}).tag(sync=True)
        error = traitlets.Unicode("").tag(sync=True)
        warnings = traitlets.List(trait=traitlets.Unicode(), default_value=[]).tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        render_nonce = traitlets.Int(0).tag(sync=True)
        active_path = traitlets.Unicode("").tag(sync=True)

        def __init__(
            self,
            source: str | Path | PiiReplacementPlan,
            *,
            df: pd.DataFrame,
            data_config: DataParameters | None = None,
            persona_backend: str = "managed",
            on_plan: PlanCallback | None = None,
            time_series: TimeSeriesParameters | None = None,
            **kwargs: Any,
        ) -> None:
            self._df = df
            self._data_config = data_config if data_config is not None else DataParameters()
            self._time_series = time_series
            self._persona_backend = persona_backend
            self._on_plan = on_plan
            yaml_text = _normalize_source(source)
            state = preview_state_from_yaml(
                yaml_text,
                df=self._df,
                data_config=self._data_config,
                persona_backend=self._persona_backend,
                time_series=self._time_series,
            )
            super().__init__(
                yaml_text=state.yaml_text,
                diagram=state.diagram,
                ranges=state.ranges,
                error=state.error or "",
                warnings=list(state.warnings),
                status=_status_for_state(state, initial=True),
                **kwargs,
            )
            self._state = state
            self._sync_plan(state)

        @property
        def current_plan(self) -> PiiReplacementPlan | None:
            """Latest plan that validated successfully, if any."""
            state = getattr(self, "_state", None)
            if isinstance(state, PreviewState) and state.valid:
                return state.plan
            return None

        def _sync_plan(self, state: PreviewState) -> None:
            if state.valid and state.plan is not None and self._on_plan is not None:
                self._on_plan(state.plan)

        def refresh(self, yaml_text: str | None = None) -> PreviewState:
            """Re-parse YAML, validate against the dataset, and refresh traits."""
            text = self.yaml_text if yaml_text is None else yaml_text
            previous = getattr(self, "_state", None)
            state = preview_state_from_yaml(
                text,
                df=self._df,
                data_config=self._data_config,
                previous=previous if isinstance(previous, PreviewState) else None,
                persona_backend=self._persona_backend,
                time_series=self._time_series,
            )
            self._state = state
            self.yaml_text = state.yaml_text
            self.diagram = state.diagram
            self.ranges = state.ranges
            self.error = state.error or ""
            self.warnings = list(state.warnings)
            self.status = _status_for_state(state)
            if state.valid:
                self._sync_plan(state)
            return state

        @traitlets.observe("render_nonce")
        def _on_render_nonce(self, change: dict[str, Any]) -> None:
            if change.get("name") != "render_nonce":
                return
            self.refresh(self.yaml_text)

    _PiiPlanPreview.__name__ = "PiiPlanPreview"
    _PiiPlanPreview.__qualname__ = "PiiPlanPreview"
    return _PiiPlanPreview


try:
    PiiPlanPreview = _build_widget_class()
except ImportError:

    class PiiPlanPreview:  # type: ignore[no-redef]
        """Notebook widget placeholder when ``notebook`` deps are missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(_NOTEBOOK_EXTRA_HINT)


def preview_pii_replacement_plan(
    source: str | Path | PiiReplacementPlan,
    *,
    df: pd.DataFrame,
    data_config: DataParameters | None = None,
    persona_backend: str = "managed",
    on_plan: PlanCallback | None = None,
    time_series: TimeSeriesParameters | None = None,
) -> Any:
    """Create a notebook widget for a PII plan validated against ``df``.

    The return type is ``Any`` because the widget class is built at import time
    only when the optional ``anywidget`` dependency is present.

    Prefer :meth:`nemo_safe_synthesizer.sdk.library_builder.SafeSynthesizer.preview_replace_pii`
    so discovery and plan sync stay inside the builder workflow.

    Args:
        source: Plan YAML string, filesystem path, or ``PiiReplacementPlan``.
        df: Dataset whose columns the plan must reference.
        data_config: Data settings used by ``validate_plan`` (e.g. group key).
        persona_backend: Persona sampler backend used for placement advisories.
        on_plan: Optional callback invoked whenever a plan validates successfully.
        time_series: Time-series settings so structural columns are rejected like
            ``resolve_plan`` / preflight.

    Returns:
        A ``PiiPlanPreview`` anywidget instance.

    Raises:
        ImportError: If the ``notebook`` optional dependency is not installed.
    """
    return PiiPlanPreview(
        source,
        df=df,
        data_config=data_config,
        persona_backend=persona_backend,
        on_plan=on_plan,
        time_series=time_series,
    )

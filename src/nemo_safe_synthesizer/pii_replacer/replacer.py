# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular PII replacer entry point."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.data import DataParameters
from ..config.pii_replacement import PiiColumnPlan, PiiEntity, PiiReplacementPlan, ReplacePiiConfig, is_person_entity
from ..observability import get_logger
from .plan import PII_REPLACEMENT_PLAN_FILENAME, plan_to_runtime, resolve_plan, save_plan_to_path, unique_id_advisories
from .replacement import run_replacement
from .runtime_config import runtime_config_from_replace_pii
from .transform_result import ColumnStatistics, TransformResult

logger = get_logger(__name__)


def _plan_column_counts(plan: PiiReplacementPlan) -> tuple[int, int, int]:
    """Return (persona_count, structured_column_count, free_text_column_count)."""
    structured = sum(
        1 for spec in plan.columns.values() if spec.entity_type and spec.entity_type != PiiEntity.free_text
    )
    free_text = sum(1 for spec in plan.columns.values() if spec.entity_type == PiiEntity.free_text)
    return len(plan.identified_personas), structured, free_text


def _replacement_plan_source(config: ReplacePiiConfig) -> str:
    if config.is_auto_discovery:
        return "auto_discovery"
    if config.plan_path:
        return str(config.plan_path)
    return "inline"


def _person_transform_method(locale: str, backend: str) -> str:
    """Human-readable persona replacement label for the evaluation report."""
    backend_labels = {
        "managed": "personas",
        "pgm": "PGM",
        "faker": "Faker",
    }
    return f"{locale} {backend_labels.get(backend, backend)}"


class TabularPiiReplacer:
    """Replace PII in tabular training data using a declarative replacement plan."""

    def __init__(
        self,
        config: ReplacePiiConfig,
        *,
        data_config: DataParameters,
        workdir: Path | str | None = None,
    ) -> None:
        self._config = config
        self._data_config = data_config
        self._workdir = Path(workdir) if workdir else None
        self._runtime = runtime_config_from_replace_pii(config)
        self.result = None
        self.elapsed_time = 0.0
        self.resolved_plan = None

    def transform_df(self, df: pd.DataFrame) -> None:
        start = time.perf_counter()
        plan_source = _replacement_plan_source(self._config)
        group_key = self._data_config.group_training_examples_by
        logger.user.info(
            "Starting tabular PII replacement",
            extra={
                "rows": len(df),
                "columns": len(df.columns),
                "person_backend": self._runtime.persona_backend,
                "locale": self._runtime.locale,
                "group_key": group_key,
                "plan_source": plan_source,
            },
        )

        plan = resolve_plan(self._config, df, data_config=self._data_config, runtime=self._runtime)
        self.resolved_plan = plan

        role_count, structured_cols, free_text_cols = _plan_column_counts(plan)
        logger.user.info(
            "Resolved PII replacement plan",
            extra={
                "group_key": plan.group_key,
                "person_roles": role_count,
                "structured_columns": structured_cols,
                "free_text_columns": free_text_cols,
            },
        )

        if plan.group_key is None:
            logger.user.warning(
                "PII replacement has no group_key; each row is treated as a distinct identity. "
                "Set data.group_training_examples_by to keep personas consistent across related rows."
            )

        for warning in unique_id_advisories(df, plan, self._runtime):
            logger.user.warning(warning)

        runtime_plan = plan_to_runtime(plan)
        replaced_df, details = run_replacement(df, runtime_plan, self._runtime)
        self.result = TransformResult(
            transformed_df=replaced_df,
            column_statistics=self._build_column_statistics(df, plan, details),
        )
        self.elapsed_time = time.perf_counter() - start

        changed_summary = details.get("changed_summary", [])
        cells_changed = sum(item["cells_changed"] for item in changed_summary)
        plan_path = None
        if self._workdir:
            plan_path = self._emit_plan(plan)

        logger.user.info(
            "",
            extra={
                "ctx": {
                    "render_table": True,
                    "tabular_data": {
                        "rows": len(df),
                        "columns_changed": len(changed_summary),
                        "cells_changed": cells_changed,
                        "person_instances": len(details.get("instances", [])),
                        "duration_sec": round(self.elapsed_time, 2),
                        "plan_path": plan_path or "not written",
                    },
                    "title": "PII Replacement Complete",
                }
            },
        )

        if plan_path is not None:
            logger.runtime.info(f"Wrote PII replacement plan to {plan_path}")

    def _emit_plan(self, plan: PiiReplacementPlan) -> Path:
        return save_plan_to_path(plan, self._workdir / PII_REPLACEMENT_PLAN_FILENAME)

    def _build_column_statistics(
        self,
        original_df: pd.DataFrame,
        plan: PiiReplacementPlan,
        details: dict[str, Any],
    ) -> dict[str, ColumnStatistics]:
        stats: dict[str, ColumnStatistics] = {}
        changed = {d["column"]: d["cells_changed"] for d in details.get("changed_summary", [])}
        locale = self._runtime.locale
        person_method = _person_transform_method(locale, self._runtime.persona_backend)

        def _add(
            col: str,
            entity: str | None,
            *,
            transformed: bool,
            transform_methods: set[str] | None = None,
        ) -> None:
            values = original_df[col].dropna().unique().tolist() if col in original_df.columns else []
            entity_key = entity or "none"
            # ``free_text`` is a plan marker for propagation-based replacement; entity
            # counts in the report list propagated PII types only, not ``free_text``.
            if entity == PiiEntity.free_text.value:
                detected_counts: dict[str, int] = {}
                detected_values: dict[str, set] = {}
            else:
                detected_counts = {entity_key: len(values)} if entity else {}
                detected_values = {entity_key: set(str(v) for v in values)} if entity else {}
            methods = transform_methods if transform_methods is not None else set()
            stats[col] = ColumnStatistics(
                assigned_type="structured" if entity != PiiEntity.free_text.value else "text",
                assigned_entity=entity,
                detected_entity_counts=detected_counts,
                detected_entity_values=detected_values,
                is_transformed=transformed,
                transform_methods=methods,
            )

        def _method_for(spec: PiiColumnPlan) -> str:
            """Human-readable transform method matching how the value is generated."""
            if spec.entity_type == PiiEntity.free_text:
                return "propagation"
            # Birth dates are age-preserving perturbations of the original date, not
            # persona/Faker draws (see core._synth_dob_programmatic).
            if spec.entity_type == PiiEntity.date_of_birth:
                return "perturbation"
            if spec.persona or is_person_entity(spec.entity_type):
                return person_method
            # Non-person entities use the inferred pattern template when available and
            # otherwise fall back to Faker (see core.build_non_person_maps).
            return "pattern" if spec.pattern else "Faker"

        for col, spec in plan.columns.items():
            entity = spec.entity_type.value if spec.entity_type else None
            methods = {_method_for(spec)} if changed.get(col, 0) > 0 else set()
            _add(col, entity, transformed=changed.get(col, 0) > 0, transform_methods=methods)

        for ent in details.get("free_text_entities", []):
            col = ent.get("column")
            label = ent.get("label")
            if not col:
                continue
            existing = stats.get(col)
            if existing is None:
                _add(col, PiiEntity.free_text.value, transformed=changed.get(col, 0) > 0)
                existing = stats[col]
            if label:
                existing.detected_entity_counts[label] = existing.detected_entity_counts.get(label, 0) + 1
                existing.detected_entity_values.setdefault(label, set()).add(str(ent.get("original")))

        return stats

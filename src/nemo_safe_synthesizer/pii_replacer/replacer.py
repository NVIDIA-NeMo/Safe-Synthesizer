# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular PII replacer entry point."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import pandas as pd

from ..config.data import DataParameters
from ..config.replace_pii import PiiColumnPlan, PiiEntity, PiiReplacementPlan, PiiReplacementScope, ReplacePiiConfig
from ..config.time_series import TimeSeriesParameters
from ..errors import InternalError
from ..observability import get_logger
from . import entities
from .llm import PiiDiscoveryEnhancer, PiiReplacementEnhancer
from .models import ReplacementOutcome
from .planning import PII_REPLACEMENT_PLAN_FILENAME, iter_plan_advisories, resolve_plan, save_plan_to_path
from .replacement import run_replacement
from .transform_result import ColumnStatistics, TransformResult

logger = get_logger(__name__)


def _plan_column_counts(plan: PiiReplacementPlan) -> tuple[int, int, int]:
    """Return (persona count, persona-backed column count, standalone column count)."""
    persona_backed = sum(len(persona.columns_to_replace) for persona in plan.persona_backed_columns)
    return len(plan.persona_backed_columns), persona_backed, len(plan.standalone_columns_to_replace)


def _replacement_plan_source(config: ReplacePiiConfig) -> str:
    if config.is_auto_discovery:
        return "auto_discovery"
    if config.plan_path:
        return str(config.plan_path)
    return "inline"


def _persona_transform_method(locale: str, backend: str) -> str:
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
        time_series: TimeSeriesParameters | None = None,
        discovery_enhancer: PiiDiscoveryEnhancer | None = None,
        replacement_enhancer: PiiReplacementEnhancer | None = None,
    ) -> None:
        self._config = config
        self._data_config = data_config
        self._time_series = time_series
        self._workdir = Path(workdir) if workdir else None
        # Injected LLM stacking providers; when None the run picks noop /
        # not-implemented from replace_pii.llm_enhancement.
        self._discovery_enhancer = discovery_enhancer
        self._replacement_enhancer = replacement_enhancer
        self._cfg = entities.config_from_replace_pii(config)
        self.result: TransformResult | None = None
        self.elapsed_time = 0.0
        self.resolved_plan: PiiReplacementPlan | None = None

    def transform_df(self, df: pd.DataFrame) -> None:
        start = time.perf_counter()
        if not df.index.is_unique:
            logger.runtime.warning(
                "Input DataFrame index is not unique; resetting to a positional index for PII replacement."
            )
            df = df.reset_index(drop=True)
        plan_source = _replacement_plan_source(self._config)
        group_key = self._data_config.group_training_examples_by
        logger.user.info(
            "Starting tabular PII replacement",
            extra={
                "rows": len(df),
                "columns": len(df.columns),
                "persona_backend": self._cfg.persona_backend,
                "locale": self._cfg.locale,
                "group_key": group_key,
                "plan_source": plan_source,
            },
        )

        plan = resolve_plan(
            self._config,
            df,
            data_config=self._data_config,
            cfg=self._cfg,
            time_series=self._time_series,
            enhancer=self._discovery_enhancer,
        )
        self.resolved_plan = plan

        persona_count, persona_backed_cols, standalone_cols = _plan_column_counts(plan)
        logger.user.info(
            "Resolved PII replacement plan",
            extra={
                "scope": plan.scope.value,
                "group_training_examples_by": group_key,
                "personas": persona_count,
                "persona_backed_columns": persona_backed_cols,
                "standalone_columns": standalone_cols,
            },
        )

        if plan.scope != PiiReplacementScope.group and group_key:
            logger.user.warning(
                f"PII replacement scope is {plan.scope.value!r} while data.group_training_examples_by is set; "
                "persona consistency follows scope, not the training group key."
            )

        for advisory in iter_plan_advisories(plan, persona_backend=self._cfg.persona_backend):
            logger.user.warning(advisory.message)

        outcome = run_replacement(df, plan, self._cfg, group_key=group_key, enhancer=self._replacement_enhancer)
        self.result = TransformResult(
            transformed_df=outcome.replaced_df,
            column_statistics=self._build_column_statistics(df, plan, outcome),
        )
        self.elapsed_time = time.perf_counter() - start

        changed_summary = cast(list[dict[str, object]], outcome.details.get("changed_summary", []))
        cells_changed = sum(int(cast(int, item["cells_changed"])) for item in changed_summary)
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
                        "persona_instances": len(outcome.instances),
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
        if self._workdir is None:
            raise InternalError("PII replacement plan emit requires a workdir")
        return save_plan_to_path(plan, self._workdir / PII_REPLACEMENT_PLAN_FILENAME)

    def _build_column_statistics(
        self,
        original_df: pd.DataFrame,
        plan: PiiReplacementPlan,
        outcome: ReplacementOutcome,
    ) -> dict[str, ColumnStatistics]:
        stats: dict[str, ColumnStatistics] = {}
        changed_raw = cast(list[dict[str, object]], outcome.details.get("changed_summary", []))
        changed = {str(d["column"]): int(cast(int, d["cells_changed"])) for d in changed_raw}
        locale = self._cfg.locale
        effective_backend = outcome.details.get("persona_backend_effective") or self._cfg.persona_backend
        persona_method = _persona_transform_method(locale, str(effective_backend))

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

        def _method_for(spec: PiiColumnPlan, *, persona_backed: bool) -> str:
            """Human-readable transform method matching how the value is generated."""
            entity_spec = entities.spec(spec.entity_type.value) if spec.entity_type else None
            if entity_spec is not None and entity_spec.transform_method:
                return entity_spec.transform_method
            path = (
                entities.effective_apply_path(spec.entity_type.value, self._cfg.persona_backend)
                if spec.entity_type
                else None
            )
            # Persona path reports the persona backend method wherever the plan puts
            # the column; standalone maps report pattern vs Faker from the plan formats.
            if persona_backed and path == "persona":
                return persona_method
            return "pattern" if spec.patterns else "Faker"

        for persona_set in plan.persona_backed_columns:
            for spec in persona_set.columns_to_replace:
                col = spec.column_name
                entity = spec.entity_type.value if spec.entity_type else None
                methods = {_method_for(spec, persona_backed=True)} if changed.get(col, 0) > 0 else set()
                _add(col, entity, transformed=changed.get(col, 0) > 0, transform_methods=methods)

        for spec in plan.standalone_columns_to_replace:
            col = spec.column_name
            entity = spec.entity_type.value if spec.entity_type else None
            methods = {_method_for(spec, persona_backed=False)} if changed.get(col, 0) > 0 else set()
            _add(col, entity, transformed=changed.get(col, 0) > 0, transform_methods=methods)

        for ent in cast(list[dict[str, object]], outcome.details.get("free_text_entities", [])):
            col = ent.get("column")
            label = ent.get("label")
            if not col:
                continue
            col_s = str(col)
            existing = stats.get(col_s)
            if existing is None:
                _add(col_s, PiiEntity.free_text.value, transformed=changed.get(col_s, 0) > 0)
                existing = stats[col_s]
            if label:
                label_s = str(label)
                existing.detected_entity_counts[label_s] = existing.detected_entity_counts.get(label_s, 0) + 1
                existing.detected_entity_values.setdefault(label_s, set()).add(str(ent.get("original")))

        return stats

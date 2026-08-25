# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-table (database-scope) PII replacer orchestrator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...config.replace_pii import AUTO_DISCOVERY, PiiReplacementPlan, PiiReplacementScope, ReplacePiiConfig
from ...defaults import DEFAULT_ARTIFACTS_PATH
from ...errors import ParameterError
from ...observability import get_logger
from .. import entities
from ..llm import PiiDiscoveryEnhancer, PiiReplacementEnhancer
from ..planning.io import PII_REPLACEMENT_PLAN_FILENAME, load_plan_from_path, save_plan_to_path
from ..replacement.apply import run_replacement
from ..replacement.personas import PersonaEngine
from .discovery import discover_database_plan
from .io import load_csv_tables, write_csv_table
from .map_io import PII_REPLACEMENT_MAP_FILENAME, save_replacement_map
from .order import processing_order
from .projection import build_table_context, project_table_plan
from .schema import DatabaseSchema, load_schema
from .store import SharedRuntimeStore

logger = get_logger(__name__)

__all__ = ["MultiTablePiiReplacer"]


def _resolve_workdir(workdir: Path | str | None) -> Path:
    if workdir is not None:
        path = Path(workdir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = Path(DEFAULT_ARTIFACTS_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


class MultiTablePiiReplacer:
    """Replace PII across a folder of related CSV tables using a shared runtime store.

    Multi-table runs are PII-only. Use ``transform_folder``; do not wire
    ``scope: database`` into ``SafeSynthesizer.run()``.
    """

    def __init__(
        self,
        config: ReplacePiiConfig,
        *,
        workdir: Path | str | None = None,
        discovery_enhancer: PiiDiscoveryEnhancer | None = None,
        replacement_enhancer: PiiReplacementEnhancer | None = None,
    ) -> None:
        self._config = config
        self._workdir = _resolve_workdir(workdir)
        self._discovery_enhancer = discovery_enhancer
        self._replacement_enhancer = replacement_enhancer
        self._cfg = entities.config_from_replace_pii(config)
        self.resolved_plan: PiiReplacementPlan | None = None
        self.store: SharedRuntimeStore | None = None
        self.transformed_tables: dict[str, pd.DataFrame] = {}

    def transform_folder(
        self,
        input_dir: Path | str,
        output_dir: Path | str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Replace PII in a folder of CSVs; persist plan, map, and transformed tables.

        Tables are processed one at a time in FK topological order. Each
        transformed CSV is written as soon as that table finishes, and the
        shared replacement map is created/updated after each table.

        Args:
            input_dir: Directory of top-level ``*.csv`` files (stems = table names).
            output_dir: Where to write transformed CSVs. Defaults to ``workdir``.

        Returns:
            Mapping of table name to transformed DataFrame.
        """
        schema = self._load_schema()
        frames = load_csv_tables(input_dir, schema)
        plan = self._resolve_plan(frames, schema)
        self.resolved_plan = plan

        if plan.scope != PiiReplacementScope.database:
            raise ParameterError(
                "MultiTablePiiReplacer requires a plan with scope 'database'; "
                f"got scope={plan.scope.value!r}. For single-table scopes use TabularPiiReplacer."
            )

        plan_path = save_plan_to_path(plan, self._workdir / PII_REPLACEMENT_PLAN_FILENAME)
        logger.user.info(f"[PII Replacement] Replacement plan written to {plan_path}")

        store = SharedRuntimeStore(
            seed=self._cfg.random_seed,
            locale=self._cfg.locale,
            key_domains=list(plan.key_domains),
        )
        self.store = store

        out_root = Path(output_dir) if output_dir is not None else self._workdir
        out_root.mkdir(parents=True, exist_ok=True)

        order = processing_order(schema)
        # One persona engine for the whole folder so managed parquet (multi-GB)
        # is loaded at most once and shared across tables.
        max_rows = max((len(df) for df in frames.values()), default=1)
        persona_engine = PersonaEngine(self._cfg, max(max_rows, 1))
        map_path = self._workdir / PII_REPLACEMENT_MAP_FILENAME
        map_exists = map_path.is_file()
        transformed: dict[str, pd.DataFrame] = {}
        for table_name in order:
            df = frames.pop(table_name)
            table_plan = plan.tables.get(table_name)
            if table_plan is None:
                logger.user.warning(
                    f"[PII Replacement] No plan section for table {table_name!r}; copying unchanged."
                )
                replaced_df = df.copy()
            else:
                logger.user.info(
                    f"[PII Replacement] Replacing PII in table {table_name!r} "
                    f"({len(df)} rows, {len(df.columns)} columns)"
                )
                projected = project_table_plan(table_name, table_plan)
                ctx = build_table_context(
                    table_name,
                    table_plan,
                    list(plan.key_domains),
                    store,
                    polymorphic_foreign_keys=plan.polymorphic_foreign_keys,
                )
                outcome = run_replacement(
                    df,
                    projected,
                    self._cfg,
                    persona_engine=persona_engine,
                    enhancer=self._replacement_enhancer,
                    store=store,
                    table_ctx=ctx,
                )
                replaced_df = outcome.replaced_df

            transformed[table_name] = replaced_df
            table_path = write_csv_table(table_name, replaced_df, out_root)
            logger.user.info(f"[PII Replacement] Transformed table {table_name!r} written to {table_path}")

            map_path = save_replacement_map(store, map_path)
            verb = "updated" if map_exists else "created"
            logger.user.info(f"[PII Replacement] Replacement map {verb} at {map_path}")
            map_exists = True

        self.transformed_tables = transformed
        logger.user.info(
            "Multi-table PII replacement complete",
            extra={
                "tables": len(transformed),
                "plan_path": str(plan_path),
                "map_path": str(map_path),
                "output_dir": str(out_root),
            },
        )
        return transformed

    def _load_schema(self) -> DatabaseSchema:
        if not self._config.schema_path:
            raise ParameterError(
                "database-scope replacement requires replace_pii.schema_path pointing to a schema YAML"
            )
        return load_schema(self._config.schema_path)

    def _resolve_plan(
        self,
        frames: dict[str, pd.DataFrame],
        schema: DatabaseSchema,
    ) -> PiiReplacementPlan:
        if self._config.is_auto_discovery:
            return discover_database_plan(
                frames,
                schema,
                self._cfg,
                self._config,
                enhancer=self._discovery_enhancer,
            )
        if self._config.plan_path:
            return load_plan_from_path(self._config.plan_path)
        inline = self._config.inline_plan
        if inline is not None:
            return inline
        raise ParameterError(
            f"replacement_plan must be {AUTO_DISCOVERY!r}, a plan path, or an inline plan"
        )

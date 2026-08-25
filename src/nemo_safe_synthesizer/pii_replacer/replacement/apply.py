# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writing the resolved replacements onto the frame, structured then free text."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable
from typing import cast

import pandas as pd

from ...config.replace_pii import PiiEntity, PiiReplacementPlan
from ...observability import get_logger
from .. import entities
from ..llm import PiiReplacementEnhancer, select_replacement_enhancer
from ..models import PersonaInstance, ReplacementOutcome, ScopedValueMap
from ..multi_table.store import SharedRuntimeStore, TableRunContext
from .free_text import build_text_substituter, instance_text_pair_labels, resolve_freetext_detections
from .instances import compute_instance_synthetics, extract_instances
from .personas import PersonaEngine
from .scope import unit_key
from .standalone import build_standalone_maps, iter_standalone_specs

logger = get_logger(__name__)

__all__ = ["apply_replacements", "run_replacement"]


def _free_text_columns(plan: PiiReplacementPlan) -> list[str]:
    cols: list[str] = []
    for col_set in plan.persona_backed_columns:
        for spec in col_set.columns_to_replace:
            if spec.entity_type == PiiEntity.free_text:
                cols.append(spec.column_name)
    for spec in plan.standalone_columns_to_replace:
        if spec.entity_type == PiiEntity.free_text:
            cols.append(spec.column_name)
    return list(dict.fromkeys(cols))


def _ingest_instances_into_store(
    df: pd.DataFrame,
    instances: list[PersonaInstance],
    store: SharedRuntimeStore,
    table_ctx: TableRunContext,
) -> None:
    """Register persona instances on the shared store under each persona's key domain."""
    for inst in instances:
        binding = table_ctx.persona_key_bindings.get(inst.persona)
        if binding is None or not inst.row_indices:
            continue
        domain_id, key_col = binding
        if key_col not in df.columns:
            continue
        for idx in inst.row_indices:
            original_key = entities.sval(df.at[idx, key_col])
            if original_key is None:
                continue
            synthetic_key = None
            if key_col in inst.synthetic_by_column:
                synthetic_key = inst.synthetic_by_column.get(key_col)
            domain_state = store.domains.get(domain_id)
            if domain_state and original_key in domain_state.values:
                synthetic_key = domain_state.values[original_key]
            store.ingest_persona_instance(
                domain_id=domain_id,
                original_key=original_key,
                synthetic_key=synthetic_key,
                originals_by_label=dict(inst.originals_by_label),
                synthetic_by_label=dict(inst.synthetic_by_label),
                free_text_pairs=list(inst.free_text_pairs),
                sex=inst.sex,
                race_raw=inst.race_raw,
            )


def _row_person_pairs_from_store(
    df: pd.DataFrame,
    store: SharedRuntimeStore,
    table_ctx: TableRunContext,
) -> dict[Hashable, list[tuple[str, str]]]:
    """Free-text pairs for rows that reference a known person via FK/PK or polymorphic Id."""
    from ..multi_table.polymorphic import resolve_polymorphic_domain

    out: dict[Hashable, list[tuple[str, str]]] = {}
    if not table_ctx.person_ref_columns and not table_ctx.polymorphic_routes:
        return out
    for idx in df.index:
        pairs: list[tuple[str, str]] = []
        # Ordinary person-reference columns
        for bare_col, domain_id in table_ctx.person_ref_columns.items():
            if bare_col in table_ctx.polymorphic_routes:
                continue  # handled below with routing
            if bare_col not in df.columns:
                continue
            original_key = entities.sval(df.at[idx, bare_col])
            if original_key is None:
                continue
            for pair in store.free_text_pairs_for_key(domain_id, original_key):
                if pair not in pairs:
                    pairs.append(pair)
        # Polymorphic person-reference columns
        for bare_col, route in table_ctx.polymorphic_routes.items():
            if bare_col not in df.columns:
                continue
            original_key = entities.sval(df.at[idx, bare_col])
            if original_key is None:
                continue
            type_value = None
            if route.type_column in df.columns:
                type_value = entities.sval(df.at[idx, route.type_column])
            domain_id = resolve_polymorphic_domain(
                store, route, original=original_key, type_value=type_value
            )
            if domain_id is None:
                continue
            state = store.domains.get(domain_id)
            if state is None or not state.person_reference:
                continue
            for pair in store.free_text_pairs_for_key(domain_id, original_key):
                if pair not in pairs:
                    pairs.append(pair)
        if pairs:
            out[idx] = pairs
    return out


def apply_replacements(
    source_df: pd.DataFrame,
    original_df: pd.DataFrame,
    instances: list[PersonaInstance],
    standalone_maps: dict[str, ScopedValueMap],
    plan: PiiReplacementPlan,
    cfg: entities.Config,
    *,
    group_key: str | None = None,
    enhancer: PiiReplacementEnhancer | None = None,
    store: SharedRuntimeStore | None = None,
    table_ctx: TableRunContext | None = None,
) -> ReplacementOutcome:
    """Apply structured and free-text replacements to a copy of the source frame.

    Structured persona and standalone maps are written first. Free-text columns
    then receive row-local substitutions built from instance pairs and standalone
    map entries. LLM detections are resolved programmatically when implemented.

    Args:
        source_df: Frame to copy and mutate (typically the same as ``original_df``).
        original_df: Unmodified source for lookups and free-text detection.
        instances: Persona instances with ``synthetic_by_column`` and ``free_text_pairs`` filled.
        standalone_maps: Per-column scoped maps from the standalone pass.
        plan: Resolved replacement plan.
        cfg: Replacement configuration.
        group_key: Training group-key column when scope is ``"group"``.
        enhancer: Optional replacement enhancer override.
        store: Optional shared multi-table runtime store.
        table_ctx: Optional per-table context for ``store``.

    Returns:
        ``ReplacementOutcome`` with ``replaced_df``, ``free_text_applied``, and
        diagnostic ``details`` (structured/standalone cols, change summary, entities).
        Callers that own instances/maps (``run_replacement``) fill those fields.
    """
    replaced_df = source_df.copy()
    structured_cols: set[str] = set()
    plan_scope = plan.scope.value
    column_order = list(source_df.columns)
    backend = cfg.persona_backend

    for inst in instances:
        if not inst.synthetic_by_column:
            continue
        for col, syn in inst.synthetic_by_column.items():
            replaced_df.loc[inst.row_indices, col] = syn
            structured_cols.add(col)

    standalone_cols: list[str] = []
    for col, cm in standalone_maps.items():
        if col not in replaced_df.columns:
            continue
        match cm.kind:
            case "flat":
                if cm.data:
                    flat_map = cast(dict[str, str], cm.data)
                    mapped = original_df[col].map(
                        lambda v, m=flat_map: m.get(sv) if (sv := entities.sval(v)) is not None else None
                    )
                    replaced_df[col] = mapped.where(mapped.notna(), replaced_df[col])
            case "group":
                for gval, raw_mapping in cm.data.items():
                    mapping = cast(dict[str, str], raw_mapping)
                    if not mapping:
                        continue
                    mask = original_df[group_key] == gval
                    mapped = original_df.loc[mask, col].map(
                        lambda v, m=mapping: m.get(sv) if (sv := entities.sval(v)) is not None else None
                    )
                    replaced_df.loc[mask, col] = mapped.where(mapped.notna(), replaced_df.loc[mask, col])
            case "record":
                for idx, raw_mapping in cm.data.items():
                    mapping = cast(dict[str, str], raw_mapping)
                    if not mapping:
                        continue
                    ov = entities.sval(original_df.at[idx, col])
                    if ov in mapping:
                        replaced_df.at[idx, col] = mapping[ov]
        structured_cols.add(col)
        standalone_cols.append(col)

    # Free-text pairs stay row-/instance-local: each instance contributes only to
    # rows it covers. Merging by group value or field signature would let one
    # persona's substitutions rewrite another row's notes (and under record scope
    # would merge competing mappings for duplicate structured identities).
    row_text_pairs: dict[Hashable, list[tuple[str, str]]] = {}
    for inst in instances:
        if not inst.free_text_pairs:
            continue
        for idx in inst.row_indices:
            existing = row_text_pairs.setdefault(idx, [])
            for pair in inst.free_text_pairs:
                if pair not in existing:
                    existing.append(pair)

    if store is not None and table_ctx is not None:
        for idx, pairs in _row_person_pairs_from_store(original_df, store, table_ctx).items():
            existing = row_text_pairs.setdefault(idx, [])
            for pair in pairs:
                if pair not in existing:
                    existing.append(pair)

    ft_cols = [c for c in _free_text_columns(plan) if c not in structured_cols and c in replaced_df.columns]

    # Structured persona/standalone mappings now exist. Detect against the
    # original text, resolve detections programmatically, and only then apply
    # all free-text substitutions to the already-structured output frame.
    llm = select_replacement_enhancer(llm_enhancement=cfg.llm_enhancement, enhancer=enhancer)
    detections = llm.detect_freetext_entities(original_df, ft_cols, plan, cfg)
    resolved_pairs = resolve_freetext_detections(
        detections,
        original_df=original_df,
        contextual_pairs=row_text_pairs,
        standalone_maps=standalone_maps,
        plan=plan,
        cfg=cfg,
        group_key=group_key,
    )
    for idx, pairs in resolved_pairs.items():
        existing = row_text_pairs.setdefault(idx, [])
        for pair in pairs:
            if pair not in existing:
                existing.append(pair)
    _PAIR_VOLUME_WARN = 500

    def _row_pairs(idx: Hashable) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = list(row_text_pairs.get(idx, []))
        for col, cm in standalone_maps.items():
            ov = entities.sval(original_df.at[idx, col]) if col in original_df.columns else None
            if ov is None:
                continue
            match cm.kind:
                case "flat":
                    mapping = cast(dict[str, str], cm.data)
                case "group":
                    mapping = cast(dict[str, str], cm.data.get(original_df.at[idx, group_key], {}))
                case "record":
                    mapping = cast(dict[str, str], cm.data.get(idx, {}))
            if ov in mapping:
                pairs.append((ov, mapping[ov]))
        if len(pairs) > _PAIR_VOLUME_WARN:
            logger.runtime.warning(
                f"[PII Replacement] Row {idx!r} has {len(pairs)} free-text substitution pairs; "
                "large pair sets slow replacement and may rewrite unrelated mentions."
            )
        return pairs

    _sub_cache: dict[tuple[tuple[str, str], ...], Callable[[object], object]] = {}
    # Pairs computed once during substitution; diagnostic accounting reuses them.
    row_pairs_by_idx: dict[Hashable, list[tuple[str, str]]] = {}

    def _row_substituter(idx: Hashable) -> Callable[[object], object] | None:
        pairs = _row_pairs(idx)
        row_pairs_by_idx[idx] = pairs
        if not pairs:
            return None
        key = tuple(sorted(set(pairs)))
        sub = _sub_cache.get(key)
        if sub is None:
            sub = build_text_substituter(list(key))
            _sub_cache[key] = sub
        return sub

    free_text_applied: list[str] = []
    free_text_entities: list[dict[str, object]] = []
    if ft_cols and (row_text_pairs or standalone_maps):
        for idx in original_df.index:
            sub = _row_substituter(idx)
            if sub is None:
                continue
            for col in ft_cols:
                val = replaced_df.at[idx, col]
                if isinstance(val, str) and val:
                    replaced_df.at[idx, col] = cast(str, sub(val))
        free_text_applied = ft_cols

        np_entity = {
            spec.column_name: spec.entity_type.value
            for spec in iter_standalone_specs(plan, backend)
            if spec.entity_type is not None
        }
        label_of: dict[str, str] = {}
        for col, cm in standalone_maps.items():
            ent = np_entity.get(col) or "unique_identifier"
            submaps = [cm.data] if cm.kind == "flat" else cm.data.values()
            for mapping in submaps:
                for ov in cast(dict[str, str], mapping):
                    label_of.setdefault(str(ov), ent)
        for inst in instances:
            for val, lab in instance_text_pair_labels(inst, cfg).items():
                label_of.setdefault(str(val), lab)

        # One compiled word-boundary pattern per distinct original; re.search with a
        # fresh concatenated string would miss the module regex cache and recompile
        # on every rows × cols × pairs check.
        needle_re: dict[str, re.Pattern[str]] = {}

        def _present(needle: str, hay: str) -> bool:
            if not needle:
                return False
            pat = needle_re.get(needle)
            if pat is None:
                pat = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", flags=re.IGNORECASE)
                needle_re[needle] = pat
            return pat.search(hay) is not None

        seen: set[tuple[str, str, str]] = set()
        for idx, pairs in row_pairs_by_idx.items():
            if not pairs:
                continue
            if plan_scope == "group" and group_key:
                scope_label = "group"
                key: Hashable | None = cast(Hashable, original_df.at[idx, group_key])
                uk = unit_key("group", key, [idx])
            elif plan_scope == "record":
                scope_label = "record"
                key = idx
                uk = unit_key("record", None, [idx])
            else:
                scope_label = "dataframe"
                key = "dataframe"
                uk = "dataframe"
            for col in ft_cols:
                otext = entities.sval(original_df.at[idx, col])
                if not otext:
                    continue
                for original, synthetic in pairs:
                    if not _present(original, otext):
                        continue
                    dedup = (str(uk), col, str(original).lower())
                    if dedup in seen:
                        continue
                    seen.add(dedup)
                    free_text_entities.append(
                        {
                            "scope": scope_label,
                            "key": key,
                            "unit_key": uk,
                            "original": original,
                            "label": label_of.get(str(original), "full_name"),
                            "synthetic": synthetic,
                            "detector": "free_text_propagation",
                            "column": col,
                        }
                    )

    def _cells_changed(col: str) -> int:
        orig_s, new_s = original_df[col].map(entities.sval), replaced_df[col].map(entities.sval)
        both_na = orig_s.isna() & new_s.isna()
        return int(((orig_s != new_s) & ~both_na).sum())

    changed_summary = [{"column": c, "cells_changed": _cells_changed(c)} for c in column_order]
    changed_summary = [d for d in changed_summary if d["cells_changed"]]

    return ReplacementOutcome(
        replaced_df=replaced_df,
        free_text_applied=free_text_applied,
        details={
            "structured_cols": structured_cols,
            "standalone_cols": standalone_cols,
            "changed_summary": changed_summary,
            "free_text_entities": free_text_entities,
        },
    )


def run_replacement(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    cfg: entities.Config,
    *,
    group_key: str | None = None,
    persona_engine: PersonaEngine | None = None,
    enhancer: PiiReplacementEnhancer | None = None,
    store: SharedRuntimeStore | None = None,
    table_ctx: TableRunContext | None = None,
) -> ReplacementOutcome:
    """Run the full PII replacement pipeline on a dataframe.

    Extracts persona instances, infers demographics, assigns synthetic personas,
    computes per-field synthetics, builds standalone maps, and applies all
    replacements (structured then free text).

    Args:
        df: Source dataframe.
        plan: Resolved replacement plan.
        cfg: Replacement configuration.
        group_key: Training group-key column when ``plan.scope`` is ``"group"``.
        persona_engine: Optional pre-built ``PersonaEngine`` (for testing).
        enhancer: Optional replacement enhancer override.
        store: Optional shared multi-table runtime store.
        table_ctx: Optional per-table context for ``store``.

    Returns:
        ``ReplacementOutcome`` with the replaced frame, persona instances,
        standalone maps, free-text columns touched, and diagnostic ``details``
        (including ``persona_backend_effective`` and change summary).
    """
    llm = select_replacement_enhancer(llm_enhancement=cfg.llm_enhancement, enhancer=enhancer)
    instances = extract_instances(df, plan, cfg, group_key=group_key)
    # Demographics inferred from names/structured context condition the
    # programmatic persona assignment that follows.
    instances = llm.infer_persona_demographics(df, instances, cfg)
    engine = persona_engine if persona_engine is not None else PersonaEngine(cfg, max(len(instances), 1))
    engine.assign(instances)
    compute_instance_synthetics(instances, cfg)
    standalone_maps = build_standalone_maps(
        df, plan, cfg, group_key=group_key, store=store, table_ctx=table_ctx
    )
    if store is not None and table_ctx is not None:
        _ingest_instances_into_store(df, instances, store, table_ctx)
    outcome = apply_replacements(
        df,
        df,
        instances,
        standalone_maps,
        plan,
        cfg,
        group_key=group_key,
        enhancer=llm,
        store=store,
        table_ctx=table_ctx,
    )
    outcome.instances = instances
    outcome.standalone_maps = standalone_maps
    outcome.details["persona_backend_effective"] = engine.backend
    return outcome

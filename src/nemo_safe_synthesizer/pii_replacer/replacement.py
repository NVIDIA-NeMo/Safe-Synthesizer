# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured and free-text replacement application."""

from __future__ import annotations

from typing import Any, NamedTuple

import pandas as pd

from ..observability import get_logger
from . import core
from .runtime_config import RuntimeConfig

logger = get_logger(__name__)


def _core_config(runtime: RuntimeConfig) -> core.Config:
    return core.Config(
        locale=runtime.locale,
        random_seed=runtime.random_seed,
        persona_backend=runtime.persona_backend,
        sdg_pgms_src=runtime.sdg_pgms_src,
        managed_assets_path=runtime.managed_assets_path,
        freetext_name_token_aliases=runtime.freetext_name_token_aliases,
        freetext_alias_min_token_len=runtime.freetext_alias_min_token_len,
        infer_value_patterns=runtime.infer_value_patterns,
        pattern_class_max=runtime.pattern_class_max,
        pattern_rare_char_frac=runtime.pattern_rare_char_frac,
        pattern_sample_cap=runtime.pattern_sample_cap,
        use_race_constraint=runtime.use_race_constraint,
        dominant_pattern_min_coverage=runtime.dominant_pattern_min_coverage,
    )


def extract_instances(df: pd.DataFrame, runtime_plan: dict[str, Any], cfg: core.Config) -> list[dict]:
    """Extract person instances using group-aware distinct-identity semantics."""
    instances: list[dict] = []
    gk = runtime_plan.get("group_key")

    for role in runtime_plan.get("roles", []):
        field_cols = {lab: c for lab, c in role.get("fields", {}).items() if c in df.columns}
        if not field_cols:
            continue
        demo = role.get("demographics") or {}
        field_meta = role.get("field_meta") or {}

        def _append(sig_rows: dict, sig_first: dict, scope: str, group_value=None) -> None:
            for _sig, idxs in sig_rows.items():
                row = sig_first[_sig]
                originals = {lab: row[c] for lab, c in field_cols.items() if pd.notna(row[c])}
                if not originals or not core._instance_is_person(field_cols, originals):
                    continue
                if scope == "group":
                    match = ("group", group_value)
                else:
                    sig_dict = {c: core._sval(row[c]) for c in field_cols.values() if pd.notna(row[c])}
                    match = ("record", sig_dict)
                inst = core._make_instance(
                    role["role"], scope, match, originals, field_cols, demo, row, cfg, field_meta
                )
                inst["group_key"] = gk
                inst["row_indices"] = list(idxs)
                instances.append(inst)

        if gk and gk in df.columns:
            for gval, gdf in df.groupby(gk, dropna=True):
                sig_rows: dict[tuple, list] = {}
                sig_first: dict[tuple, pd.Series] = {}
                for idx, row in gdf.iterrows():
                    sig = tuple((c, core._sval(row[c])) for c in field_cols.values())
                    if all(v is None for _, v in sig):
                        continue
                    sig_rows.setdefault(sig, []).append(idx)
                    sig_first.setdefault(sig, row)
                _append(sig_rows, sig_first, "group", gval)
        else:
            for idx, row in df.iterrows():
                originals = {lab: row[c] for lab, c in field_cols.items() if pd.notna(row[c])}
                if not originals or not core._instance_is_person(field_cols, originals):
                    continue
                sig_dict = {c: core._sval(row[c]) for c in field_cols.values() if pd.notna(row[c])}
                match = ("record", sig_dict)
                inst = core._make_instance(
                    role["role"], "record", match, originals, field_cols, demo, row, cfg, field_meta
                )
                inst["group_key"] = gk
                inst["row_indices"] = [idx]
                instances.append(inst)

    return instances


class NonPersonColMap(NamedTuple):
    """A non-person column's replacement map, tagged with how it is keyed.

    ``kind == "flat"``  -> ``data`` is ``{original: synthetic}`` applied to the whole
    column (used for identifiers, whose distinct values already make scoping moot).
    ``kind == "group"`` -> ``data`` is ``{group_value: {original: synthetic}}``.
    ``kind == "record"`` -> ``data`` is ``{row_index: {original: synthetic}}``.
    """

    kind: str
    data: dict


def build_non_person_maps(
    original_df: pd.DataFrame,
    runtime_plan: dict[str, Any],
    cfg: core.Config,
) -> dict[str, NonPersonColMap]:
    """Build a replacement map per non-person column.

    Identifier entities (``core.NON_PERSON_ENTITIES``) get a single value-keyed map
    built once over the column's distinct values (one seeded Faker, then a vectorized
    ``Series.map`` at apply time). This is fast and honours the consistency rule --
    a value always maps to the same synthetic, so a group-constant id is consistent
    within its group and a record-level id per record. Global uniqueness is guaranteed
    by a column-level ``used`` set seeded with every real original.

    ``date_of_birth`` is special-cased to the scoped path: it is regenerated by
    age-preserving perturbation and its values legitimately repeat, so each scope unit
    (group when a ``group_key`` is present, else record) seeds its own RNG and is
    perturbed independently -- two people sharing a birthday may get different
    synthetic dates across groups, while a date is stable within its unit.

    Temporal identify-only columns pass through unchanged.
    """
    gk = runtime_plan.get("group_key")
    maps: dict[str, NonPersonColMap] = {}

    for ent in runtime_plan.get("non_person", []):
        col, entity = ent["column"], ent["entity"]
        if col not in original_df.columns:
            continue
        # Temporal columns are identified only to keep them out of free-text; they
        # are never value-replaced.
        if entity in core.IDENTIFIED_NOT_REPLACED_ENTITIES:
            maps[col] = NonPersonColMap("flat", {})
            logger.runtime.debug(
                f"[PII Replacement] Temporal column {col!r} passes through unchanged (entity={entity})"
            )
            continue

        pattern = ent.get("pattern")
        coverage = ent.get("dominant_pattern_coverage")

        if entity == "date_of_birth":
            maps[col] = _build_dob_map(original_df, col, pattern, coverage, gk, cfg)
            continue

        maps[col] = NonPersonColMap("flat", _build_identifier_map(original_df, col, entity, pattern, cfg))

    return maps


def _build_identifier_map(
    original_df: pd.DataFrame,
    col: str,
    entity: str,
    pattern: str | None,
    cfg: core.Config,
) -> dict[str, str]:
    """One globally-unique value-keyed map over a column's distinct values."""
    fake = core._seeded_faker(cfg.random_seed ^ core._stable_hash(col), cfg.locale)
    rng = fake.random
    originals = [str(v) for v in original_df[col].dropna().unique()]
    # Seed ``used`` with every real original so a synthetic never equals a real id.
    used: set[str] = set(originals)
    mapping: dict[str, str] = {}
    for sv in originals:
        new = _unique_synthetic(sv, entity, pattern, rng, fake, used)
        if new and new != sv:
            mapping[sv] = new
            used.add(new)
    return mapping


def _build_dob_map(
    original_df: pd.DataFrame,
    col: str,
    pattern: str | None,
    coverage: float | None,
    gk: str | None,
    cfg: core.Config,
) -> NonPersonColMap:
    """Scoped birth-date map: each scope unit is perturbed with its own seed."""
    # Whole-column reuse of the dominant strftime format at 100% coverage; else
    # per-value format detection preserves minority formats.
    fmt = pattern if coverage == 100.0 else None
    fake = core._seeded_faker(cfg.random_seed ^ core._stable_hash(col), cfg.locale)
    rng = fake.random

    def _unit_map(values: pd.Series, scope_key: Any) -> dict[str, str]:
        rng.seed(cfg.random_seed ^ core._stable_hash(f"{col}\x00{scope_key}"))
        mapping: dict[str, str] = {}
        for sv in (str(v) for v in values.dropna().unique()):
            new = core._synth_dob_programmatic(sv, rng, fmt=fmt)
            if new and new != sv:
                mapping[sv] = new
        return mapping

    if gk and gk in original_df.columns:
        data = {gval: _unit_map(gdf[col], gval) for gval, gdf in original_df.groupby(gk, dropna=True)}
        return NonPersonColMap("group", data)
    data = {idx: _unit_map(original_df.loc[[idx], col], idx) for idx in original_df.index}
    return NonPersonColMap("record", data)


def _unique_synthetic(
    sv: str,
    entity: str,
    pattern: str | None,
    rng,
    fake,
    used: set[str],
) -> str | None:
    """Generate one synthetic value for ``sv`` (pattern template, else entity Faker).

    The result is guaranteed distinct from ``sv`` and from every value already in
    ``used`` (global uniqueness), with a deterministic suffix fallback for the
    pathological case where the generator's value space is exhausted.
    """

    def _fresh(cand: str | None) -> bool:
        return bool(cand) and cand != sv and cand not in used

    generators = []
    if pattern:
        generators.append(lambda: core.generate_from_pattern(pattern, rng))
    generators.append(lambda: core._fake_value(entity, sv, fake))
    for gen in generators:
        cand = gen()
        for _ in range(200):
            if _fresh(cand):
                return cand
            cand = gen()
    # Pathological space exhaustion: deterministically disambiguate.
    base = core._fake_value(entity, sv, fake) or sv
    for suffix in range(1, 100000):
        cand = f"{base}-{suffix}"
        if _fresh(cand):
            return cand
    return None


def apply_replacements(
    source_df: pd.DataFrame,
    original_df: pd.DataFrame,
    instances: list[dict],
    non_person_maps: dict[str, NonPersonColMap],
    runtime_plan: dict[str, Any],
    runtime: RuntimeConfig,
) -> dict[str, Any]:
    cfg = _core_config(runtime)
    replaced_df = source_df.copy()
    structured_cols: set[str] = set()
    group_key = runtime_plan.get("group_key")
    column_order = list(source_df.columns)

    for inst in instances:
        syn_by_col = inst.get("syn_by_col", {})
        if not syn_by_col:
            continue
        for col, syn in syn_by_col.items():
            replaced_df.loc[inst["row_indices"], col] = syn
            structured_cols.add(col)

    non_person_cols: list[str] = []
    for col, cm in non_person_maps.items():
        if col not in replaced_df.columns:
            continue
        if cm.kind == "flat":
            if cm.data:
                mapped = original_df[col].map(lambda v: cm.data.get(core._sval(v)))
                replaced_df[col] = mapped.where(mapped.notna(), replaced_df[col])
        elif cm.kind == "group":
            for gval, mapping in cm.data.items():
                if not mapping:
                    continue
                mask = original_df[group_key] == gval
                mapped = original_df.loc[mask, col].map(lambda v: mapping.get(core._sval(v)))
                replaced_df.loc[mask, col] = mapped.where(mapped.notna(), replaced_df.loc[mask, col])
        else:  # record
            for idx, mapping in cm.data.items():
                if not mapping:
                    continue
                ov = core._sval(original_df.at[idx, col])
                if ov in mapping:
                    replaced_df.at[idx, col] = mapping[ov]
        structured_cols.add(col)
        non_person_cols.append(col)

    group_text_pairs: dict[Any, list[tuple[str, str]]] = {}
    record_text_by_cols: dict[tuple[str, ...], dict[tuple, list[tuple[str, str]]]] = {}
    for inst in instances:
        pairs = inst.get("text_pairs") or []
        if not pairs:
            continue
        if inst["match"][0] == "group":
            gval = inst["match"][1]
            existing = group_text_pairs.setdefault(gval, [])
            for pair in pairs:
                if pair not in existing:
                    existing.append(pair)
        else:
            sig = inst["match"][1]
            cols = tuple(sorted(sig))
            record_text_by_cols.setdefault(cols, {}).setdefault(tuple(sig[c] for c in cols), []).extend(pairs)

    ft_cols = [
        c for c in runtime_plan.get("free_text_columns", []) if c not in structured_cols and c in replaced_df.columns
    ]

    def _row_pairs(idx) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        if group_key:
            gp = group_text_pairs.get(original_df.at[idx, group_key])
            if gp:
                pairs += gp
        for cols, lookup in record_text_by_cols.items():
            rp = lookup.get(tuple(core._sval(original_df.at[idx, c]) for c in cols))
            if rp:
                pairs += rp
        for col, cm in non_person_maps.items():
            ov = core._sval(original_df.at[idx, col]) if col in original_df.columns else None
            if ov is None:
                continue
            if cm.kind == "flat":
                mapping = cm.data
            elif cm.kind == "group":
                mapping = cm.data.get(original_df.at[idx, group_key], {})
            else:  # record
                mapping = cm.data.get(idx, {})
            if ov in mapping:
                pairs.append((ov, mapping[ov]))
        return pairs

    _sub_cache: dict[tuple, Any] = {}

    def _row_substituter(idx):
        pairs = _row_pairs(idx)
        if not pairs:
            return None
        key = tuple(sorted(set(pairs)))
        sub = _sub_cache.get(key)
        if sub is None:
            sub = core.build_text_substituter(list(key))
            _sub_cache[key] = sub
        return sub

    free_text_applied: list[str] = []
    free_text_entities: list[dict] = []
    if ft_cols and (group_text_pairs or record_text_by_cols or non_person_maps):
        for idx in original_df.index:
            sub = _row_substituter(idx)
            if sub is None:
                continue
            for col in ft_cols:
                val = replaced_df.at[idx, col]
                if isinstance(val, str) and val:
                    replaced_df.at[idx, col] = sub(val)
        free_text_applied = ft_cols

        np_entity = {e["column"]: e.get("entity") for e in runtime_plan.get("non_person", []) if e.get("column")}
        label_of: dict[str, str] = {}
        for col, cm in non_person_maps.items():
            ent = np_entity.get(col) or "unique_identifier"
            submaps = [cm.data] if cm.kind == "flat" else cm.data.values()
            for mapping in submaps:
                for ov in mapping:
                    label_of.setdefault(str(ov), ent)
        for inst in instances:
            for val, lab in core.instance_text_pair_labels(inst, cfg).items():
                label_of.setdefault(str(val), lab)

        def _present(needle: str, hay: str) -> bool:
            if not needle:
                return False
            import re

            return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", hay) is not None

        seen: set = set()
        for idx in original_df.index:
            pairs = _row_pairs(idx)
            if not pairs:
                continue
            scope = "group" if group_key else "record"
            key = original_df.at[idx, group_key] if group_key else idx
            uk = core._unit_key(scope, key if scope == "group" else None, [idx])
            for col in ft_cols:
                otext = core._sval(original_df.at[idx, col])
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
                            "scope": scope,
                            "key": key,
                            "unit_key": uk,
                            "original": original,
                            "label": label_of.get(str(original), "full_name"),
                            "synthetic": synthetic,
                            "detector": "mvp_propagation",
                            "column": col,
                        }
                    )

    def _cells_changed(col: str) -> int:
        orig_s, new_s = original_df[col].map(core._sval), replaced_df[col].map(core._sval)
        both_na = orig_s.isna() & new_s.isna()
        return int(((orig_s != new_s) & ~both_na).sum())

    changed_summary = [{"column": c, "cells_changed": _cells_changed(c)} for c in column_order]
    changed_summary = [d for d in changed_summary if d["cells_changed"]]

    return {
        "replaced_df": replaced_df,
        "structured_cols": structured_cols,
        "free_text_applied": free_text_applied,
        "non_person_cols": non_person_cols,
        "changed_summary": changed_summary,
        "free_text_entities": free_text_entities,
    }


def run_replacement(
    df: pd.DataFrame,
    runtime_plan: dict[str, Any],
    runtime: RuntimeConfig,
    *,
    persona_engine: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = _core_config(runtime)
    instances = extract_instances(df, runtime_plan, cfg)
    engine = persona_engine
    if engine is None:
        from .persona import PersonaEngine

        engine = PersonaEngine(runtime, max(len(instances), 1))
    engine.assign(instances)
    core.compute_instance_synthetics(instances, cfg)
    non_person_maps = build_non_person_maps(df, runtime_plan, cfg)
    result = apply_replacements(df, df, instances, non_person_maps, runtime_plan, runtime)
    return result["replaced_df"], {
        "instances": instances,
        "non_person_maps": non_person_maps,
        **result,
    }

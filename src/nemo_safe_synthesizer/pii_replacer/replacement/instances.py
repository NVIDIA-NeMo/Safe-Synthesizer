# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Person instances: who the plan finds in the frame, and what they become."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pandas as pd

from ...config.replace_pii import PersonaColumnSet, PiiEntity, PiiReplacementPlan
from ...errors import InternalError, ParameterError
from .. import entities
from ..detection import looks_like_person_name
from ..models import PersonaInstance
from .demographics import norm_sex, persona_match_map, race_to_sfv
from .free_text import instance_text_pairs
from .personas import given_name, synth_value, wants_middle_name
from .scope import seeded_faker, stable_hash

__all__ = ["compute_instance_synthetics", "extract_instances"]


def _instance_is_person(field_cols: dict, originals: dict) -> bool:
    if "full_name" in field_cols and "full_name" in originals:
        return looks_like_person_name(originals["full_name"])
    return True


def _make_instance(
    persona, match, originals, field_cols, match_persona_by, row, cfg, patterns_by_label=None
) -> PersonaInstance:
    """Build one persona instance from row values and matching constraints.

    ``persona`` is the plan-level persona name (for example ``"patient"``). The
    synthetic identity chosen for this instance is attached later by
    ``PersonaEngine`` under ``synthetic_person``.

    Args:
        persona: Plan-level persona name.
        match: Scope match tuple (``("group", gval)`` or ``("record", sig)``).
        originals: Original field values for this instance.
        field_cols: Label-to-column mapping for persona-sourced fields.
        match_persona_by: Plan conditions for demographic matching.
        row: Representative row for this instance.
        cfg: Replacement configuration.
        patterns_by_label: Label-to-format-templates from ``PiiColumnPlan.patterns``.

    Returns:
        A new ``PersonaInstance`` with demographics and originals populated.
    """
    cond = persona_match_map(match_persona_by)
    sex = norm_sex(row[cond["sex"]]) if cond.get("sex") else None
    # Persona name realism depends on sex + ethnic_background for PGM/managed.
    # Faker only conditions on sex.
    race_val = None
    if (
        cfg.persona_backend != "faker"
        and cond.get("ethnic_background")
        and pd.notna(row.get(cond["ethnic_background"]))
    ):
        race_val = row[cond["ethnic_background"]]
    return PersonaInstance(
        persona=persona,
        match=match,
        field_cols=dict(field_cols),
        patterns_by_label={lab: list(pats) for lab, pats in (patterns_by_label or {}).items()},
        originals_by_label={lab: str(v) for lab, v in originals.items()},
        sex=sex,
        # Raw race value is kept for audit output. Matching uses select_field_values,
        # set programmatically by ethnicity_to_pgm.
        race_raw=entities.sval(race_val),
        select_field_values=race_to_sfv(race_val, cfg),
    )


def _persona_sourced_fields(
    col_set: PersonaColumnSet,
    persona_backend: str,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Collect persona-sourced label→column and pattern mappings.

    Skips free-text columns and labels whose effective apply path is not ``"persona"``.

    Args:
        col_set: Persona column set from the plan.
        persona_backend: Effective persona backend.

    Returns:
        Tuple of ``(field_cols, patterns_by_label)``.
    """
    fields: dict[str, str] = {}
    patterns_by_label: dict[str, list[str]] = {}
    for spec in col_set.columns_to_replace:
        if spec.entity_type is None or spec.entity_type == PiiEntity.free_text:
            continue
        label = spec.entity_type.value
        if entities.effective_apply_path(label, persona_backend) != "persona":
            continue
        fields[label] = spec.column_name
        if spec.patterns:
            patterns_by_label[label] = list(spec.patterns)
    return fields, patterns_by_label


def extract_instances(
    df: pd.DataFrame,
    plan: PiiReplacementPlan,
    cfg: entities.Config,
    *,
    group_key: str | None = None,
) -> list[PersonaInstance]:
    """Extract person instances using plan scope and optional training group key.

    Under ``scope="group"``, the same ``(fname, lname)`` within a patient group
    becomes one instance covering many ``row_indices``. Under ``scope="record"``,
    each row is its own instance.

    Structural grain (group-constant vs record-varying) partitions fields so a
    constant ``full_name`` and varying ``email`` under one plan persona do not
    share instance identity — otherwise one group would mint a new synthetic
    name per distinct email.

    Args:
        df: Source dataframe.
        plan: Resolved replacement plan.
        cfg: Replacement configuration.
        group_key: Training group-key column when ``plan.scope`` is ``"group"``.

    Returns:
        List of ``PersonaInstance`` objects (synthetic persona attached later).

    Raises:
        ParameterError: If scope is ``"group"`` but ``group_key`` is missing.
        InternalError: If group-scoped extraction reaches groupby without a key.
    """
    from ..detection.stats import scoped_column_stats

    instances: list[PersonaInstance] = []
    scope = plan.scope.value
    gk = group_key
    backend = cfg.persona_backend
    grain_stats = scoped_column_stats(df, gk if scope == "group" else None, cfg.group_constancy_threshold)
    grain_by_col = {col: str(col_stats.get("grain", "record")) for col, col_stats in grain_stats.items()}

    if scope == "group" and (not gk or gk not in df.columns):
        raise ParameterError(
            "replacement scope is 'group' but data.group_training_examples_by is not set or missing from dataframe"
        )

    def _field_partitions(field_cols: dict[str, str]) -> list[dict[str, str]]:
        """Split persona fields by structural grain when both strata are present."""
        group_fields = {lab: col for lab, col in field_cols.items() if grain_by_col.get(col) in {"group", "key"}}
        record_fields = {lab: col for lab, col in field_cols.items() if grain_by_col.get(col) == "record"}
        if group_fields and record_fields:
            return [group_fields, record_fields]
        return [field_cols]

    for col_set in plan.persona_backed_columns:
        fields, patterns_by_label = _persona_sourced_fields(col_set, backend)
        all_field_cols = {lab: c for lab, c in fields.items() if c in df.columns}
        if not all_field_cols:
            continue
        match_persona_by = [
            {"persona_attribute": cond.persona_attribute, "column_name": cond.column_name}
            for cond in col_set.match_persona_by
        ]

        for field_cols in _field_partitions(all_field_cols):
            part_patterns = {lab: pats for lab, pats in patterns_by_label.items() if lab in field_cols}

            def _append_instance(
                match: tuple,
                row: pd.Series,
                originals: dict,
                row_indices: list,
                *,
                _field_cols: dict[str, str] = field_cols,
                _patterns: dict[str, list[str]] = part_patterns,
            ) -> None:
                if not originals or not _instance_is_person(_field_cols, originals):
                    return
                inst = _make_instance(
                    col_set.persona,
                    match,
                    originals,
                    _field_cols,
                    match_persona_by,
                    row,
                    cfg,
                    _patterns,
                )
                inst.group_key = gk
                inst.row_indices = row_indices
                instances.append(inst)

            match scope:
                case "group":
                    if gk is None:
                        raise InternalError(
                            "group-scoped persona extraction reached groupby without a group key; "
                            "validate_plan should have rejected this configuration"
                        )
                    for gval, gdf in df.groupby(gk, dropna=True):
                        sig_rows: dict[tuple, list] = {}
                        sig_first: dict[tuple, pd.Series] = {}
                        for idx, row in gdf.iterrows():
                            sig = tuple((c, entities.sval(row[c])) for c in field_cols.values())
                            if all(v is None for _, v in sig):
                                continue
                            sig_rows.setdefault(sig, []).append(idx)
                            sig_first.setdefault(sig, row)
                        for _sig, idxs in sig_rows.items():
                            row = sig_first[_sig]
                            originals = {lab: row[c] for lab, c in field_cols.items() if pd.notna(row[c])}
                            _append_instance(("group", gval), row, originals, list(idxs))
                case "record":
                    for idx, row in df.iterrows():
                        originals = {lab: row[c] for lab, c in field_cols.items() if pd.notna(row[c])}
                        sig_dict = {c: entities.sval(row[c]) for c in field_cols.values() if pd.notna(row[c])}
                        _append_instance(("record", sig_dict), row, originals, [idx])
                case _:
                    sig_rows = {}
                    sig_first = {}
                    for idx, row in df.iterrows():
                        sig = tuple((c, entities.sval(row[c])) for c in field_cols.values())
                        if all(v is None for _, v in sig):
                            continue
                        sig_rows.setdefault(sig, []).append(idx)
                        sig_first.setdefault(sig, row)
                    for sig, idxs in sig_rows.items():
                        row = sig_first[sig]
                        originals = {lab: row[c] for lab, c in field_cols.items() if pd.notna(row[c])}
                        sig_dict = {c: entities.sval(row[c]) for c in field_cols.values() if pd.notna(row[c])}
                        _append_instance(("record", sig_dict), row, originals, list(idxs))

    return instances


def _person_key(inst: PersonaInstance) -> object:
    kind, payload = inst.match
    base = payload if kind == "group" else dict(cast(Mapping[str, object], payload))
    # Under group scope the match payload is the group value. Record-varying
    # fields partitioned into their own instances still share that payload, so
    # fold originals into the key or every email in the group would reuse one seed.
    originals = tuple(sorted(inst.originals_by_label.items()))
    return (base, originals)


def compute_instance_synthetics(instances: list[PersonaInstance], cfg: entities.Config) -> None:
    """Fill each instance's label/column synthetics and free-text pairs in place.

    Args:
        instances: Persona instances with ``synthetic_person`` already assigned.
        cfg: Replacement configuration (seed, locale, alias settings).
    """
    for inst in instances:
        persona = inst.synthetic_person
        synthetic_by_label: dict[str, str] = {}
        synthetic_by_column: dict[str, str] = {}
        if persona:
            seed = cfg.random_seed ^ stable_hash(str(_person_key(inst)))
            fake = seeded_faker(seed, cfg.locale)
            # No persona source carries a middle name, so one is drawn here rather
            # than per column: a person's middle name is the same wherever it is
            # written, its own column and the '{M}' of a full-name pattern alike.
            if not persona.get("middle_name") and wants_middle_name(inst):
                persona = {**persona, "middle_name": given_name(persona, fake)}
            patterns_by_label = inst.patterns_by_label
            for label, col in inst.field_cols.items():
                original = inst.originals_by_label.get(label)
                if original is None:
                    continue
                sv = synth_value(label, original, persona, fake, patterns_by_label.get(label), inst.originals_by_label)
                if sv is None or str(sv) == str(original):
                    continue
                synthetic_by_label[label] = str(sv)
                synthetic_by_column[col] = str(sv)
        inst.synthetic_by_label = synthetic_by_label
        inst.synthetic_by_column = synthetic_by_column
        inst.free_text_pairs = instance_text_pairs(inst, cfg)

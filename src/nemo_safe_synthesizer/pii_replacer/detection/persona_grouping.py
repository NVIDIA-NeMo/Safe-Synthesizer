# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona role grouping, name agreement, and structured-column orchestration."""

from __future__ import annotations

import re
from typing import cast

import pandas as pd

from ...observability import get_logger
from ..entities import (
    DEMO_LABEL_PATTERNS,
    ENTITY_NAME_PATTERNS,
    ROLE_STRIP_TOKENS,
    Config,
    demo_keys_for_backend,
    effective_apply_path,
    is_identify_only,
    is_missing_value,
    spec,
)
from ..models import ColumnEvidence
from ..patterns import date_patterns, split_full_name, value_patterns
from .column_names import (
    match_column_header,
    name_supports_value_entity,
    normalize_column_name_for_match,
)
from .free_text import detect_free_text_columns
from .value_recognizers import (
    analyze_column_patterns,
    looks_like_sequential_integer_id,
    probe_numeric_column,
    sample_looks_like_api_key,
    sample_looks_like_multi_person,
    sample_looks_like_org_name,
    sample_looks_like_street_address,
)

logger = get_logger(__name__)

# Only these name parts participate in cross-column agreement checks.
_NAME_AGREE_LABELS = frozenset({"first_name", "last_name", "full_name"})
_NAME_AGREE_THRESHOLD = 0.85
_NAME_AGREE_MIN_COMPARABLE = 3


def _persona_role_key(col: str) -> str:
    """Column-derived persona role key, or empty string when none is identifiable.

    ``patient_first_name`` → ``patient``, ``provider_email`` → ``provider``,
    ``emergency_contact_name`` → ``emergency_contact``, bare ``first_name`` → empty.

    Suffix tokens come from ``ROLE_STRIP_TOKENS`` (entity-label lexicon in
    ``entities.py``), not from header-match regexes that also list role words.
    """
    name = normalize_column_name_for_match(col)
    tokens = [t for t in re.split(r"[_\-\s]+", name) if t]
    kept = [t for t in tokens if t not in ROLE_STRIP_TOKENS]
    return "_".join(kept)


def _norm_name_token(value: object) -> str | None:
    if is_missing_value(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    return text or None


def _name_parts_agree(existing: dict[str, str], new_label: str, new_val: object, row: pd.Series) -> bool | None:
    """Whether one row's new name value agrees with existing persona name columns.

    Returns ``None`` when the row is not comparable (missing values).
    """
    new_norm = _norm_name_token(new_val)
    if new_norm is None:
        return None

    existing_labels = {label for label in existing if label in _NAME_AGREE_LABELS}
    if not existing_labels:
        return True

    # first then last only (no full yet): no cross-check until a full_name arrives.
    if new_label in {"first_name", "last_name"} and existing_labels <= {"first_name", "last_name"}:
        return True

    if new_label == "full_name":
        parts = split_full_name(str(new_val))
        first = last = None
        if "first_name" in existing:
            first = _norm_name_token(row.get(existing["first_name"]))
            if first is None:
                return None
        if "last_name" in existing:
            last = _norm_name_token(row.get(existing["last_name"]))
            if last is None:
                return None
        if first is None and last is None:
            return True
        part_ok = True
        if first is not None and first != _norm_name_token(parts.get("first_name")):
            part_ok = False
        if last is not None and last != _norm_name_token(parts.get("last_name")):
            part_ok = False
        if part_ok:
            return True
        if first is not None and last is not None:
            return new_norm in {f"{first} {last}", f"{last}, {first}", f"{last},{first}"}
        return False

    if "full_name" in existing and new_label in {"first_name", "last_name"}:
        full_raw = row.get(existing["full_name"])
        if _norm_name_token(full_raw) is None:
            return None
        parts = split_full_name(str(full_raw))
        expected = _norm_name_token(parts.get(new_label))
        return expected is not None and expected == new_norm

    return True


def _persona_names_agree(
    df: pd.DataFrame,
    existing_fields: dict[str, str],
    new_col: str,
    new_label: str,
    *,
    sample: int = 200,
) -> bool:
    """True when ``new_col`` may join a persona that already has name parts.

    Scans at most ``sample`` rows (same bound as other content gates in this module).
    """
    if new_label not in _NAME_AGREE_LABELS:
        return True
    existing_name = {label: col for label, col in existing_fields.items() if label in _NAME_AGREE_LABELS}
    if not existing_name:
        return True
    # first + last only, adding the other part: no check yet.
    if (
        new_label in {"first_name", "last_name"}
        and existing_name.keys() <= {"first_name", "last_name"}
        and new_label not in existing_name
        and "full_name" not in existing_name
    ):
        return True

    rows = df if len(df) <= sample else df.sample(sample, random_state=0)
    comparable = 0
    agree = 0
    for _, row in rows.iterrows():
        result = _name_parts_agree(existing_name, new_label, row.get(new_col), row)
        if result is None:
            continue
        comparable += 1
        if result:
            agree += 1
    if comparable < _NAME_AGREE_MIN_COMPARABLE:
        return False
    return (agree / comparable) >= _NAME_AGREE_THRESHOLD


def skip_reason_named_column(
    entity_spec,
    series: pd.Series,
    value_entity: str | None,
    apply_path: str,
) -> str | None:
    """Return a skip reason for a name-matched column, or ``None`` to allocate it.

    Example:
        Header ``phone_number`` whose values are not phones ->
        ``"values do not look like phone numbers"``.

    Args:
        entity_spec: Registry entry for the name-matched entity.
        series: Column values used for content gates.
        value_entity: Dominant value-derived entity label, or ``None``.
        apply_path: Resolved apply path (``persona`` or ``standalone_map``).

    Returns:
        Human-readable skip reason, or ``None`` when the column may be allocated.
    """
    label = entity_spec.label
    if entity_spec.requires_value_match and value_entity != label:
        if label == "phone_number":
            return "values do not look like phone numbers"
        return "values do not match that entity"
    if label == "date_of_birth" and value_entity != "date":
        return "values are not parseable dates"
    if label == "unique_identifier" and apply_path == "standalone_map" and looks_like_sequential_integer_id(series):
        return "looks like a sequential integer id (1, 2, 3, …); not treated as a unique identifier"
    if label == "api_key" and (pd.api.types.is_numeric_dtype(series) or not sample_looks_like_api_key(series)):
        return "content is numeric or not credential-like"
    if entity_spec.name_shape_gates:
        if sample_looks_like_multi_person(series):
            return (
                "looks like multi-person values (delimiters such as 'and', '/', '&'); "
                "not auto-assigned — pre-split or hand-plan"
            )
        if sample_looks_like_org_name(series):
            return "values look like organizations, not people"
    if label == "street_address" and not sample_looks_like_street_address(series):
        return "values lack house numbers (street name only)"
    return None


def detect_structured_columns(df_subset: pd.DataFrame, stats: dict, cfg: Config) -> dict:
    """Detect persona-backed and standalone PII columns over a column subset.

    Per column: gather name/value evidence, resolve ``EntitySpec`` and apply path,
    run content gates, then allocate to persona, standalone, or identify-only.

    Args:
        df_subset: Dataframe slice whose columns are classified.
        stats: Per-column stats from ``scoped_column_stats`` or ``column_stats``.
        cfg: Engine configuration controlling thresholds and persona backend.

    Returns:
        Dict with ``personas``, ``free_text_columns``, ``standalone_columns``, and
        ``identified_not_replaced`` keys.
    """
    backend = cfg.persona_backend
    fields_by_persona: dict[str, dict[str, str]] = {}
    field_patterns_by_persona: dict[str, dict[str, list[str]]] = {}
    demo_by_persona: dict[str, dict[str, str]] = {}
    role_personas: dict[str, list[str]] = {}
    empty_persona_seq = 0
    role_instance_count: dict[str, int] = {}
    standalone: list[dict] = []
    identified_not_replaced: list[str] = []
    consumed: set[str] = set()

    def _mint_persona(role: str) -> str:
        nonlocal empty_persona_seq
        if not role:
            empty_persona_seq += 1
            return f"person_{empty_persona_seq}"
        n = role_instance_count.get(role, 0) + 1
        role_instance_count[role] = n
        return role if n == 1 else f"{role}_{n}"

    def _allocate_persona(
        col: str,
        label: str,
        *,
        warn_collision: bool,
        require_name_agreement: bool = False,
    ) -> str:
        role = _persona_role_key(col)
        pool = role_personas.setdefault(role, [])
        disagreement: str | None = None
        for pid in pool:
            if label in fields_by_persona.get(pid, {}) or label in demo_by_persona.get(pid, {}):
                continue
            if require_name_agreement and label in _NAME_AGREE_LABELS:
                existing = fields_by_persona.get(pid, {})
                if any(lbl in _NAME_AGREE_LABELS for lbl in existing) and not _persona_names_agree(
                    df_subset, existing, col, label
                ):
                    disagreement = pid
                    continue
            return pid
        if disagreement is not None:
            logger.user.warning(
                f"[PII Replacement] Column {col!r} ({label}) does not agree with name columns on "
                f"persona {disagreement!r}; assigning a new persona (review the plan)."
            )
        elif warn_collision and pool:
            prior = pool[-1]
            prior_col = fields_by_persona.get(prior, {}).get(label) or demo_by_persona.get(prior, {}).get(label)
            logger.user.warning(
                f"[PII Replacement] Column {col!r} shares entity {label!r} with {prior_col!r}; "
                f"assigning a new persona so both are replaced (review the plan)."
            )
        pid = _mint_persona(role)
        pool.append(pid)
        return pid

    def _allocate_demo(col: str, label: str) -> str:
        """Attach an unprefixed demographic to the earliest compatible persona.

        Prefixed demos (``patient_sex``) stay in their role pool. Bare ``sex`` /
        ``race`` columns are dataset-level attributes and join the first persona
        that does not already have that demographic, so they can condition the
        subject persona (e.g. ``patient``) even when role keys differ.
        """
        role = _persona_role_key(col)
        if role:
            return _allocate_persona(col, label, warn_collision=False)
        for pool in role_personas.values():
            for pid in pool:
                if label not in demo_by_persona.get(pid, {}) and label not in fields_by_persona.get(pid, {}):
                    return pid
        return _allocate_persona(col, label, warn_collision=False)

    def _add_standalone(col: str, entity: str, patterns: list[str], *, note: str = "") -> None:
        standalone.append({"column": col, "entity": entity, "patterns": patterns})
        consumed.add(col)
        extra = f" — {note}" if note else ""
        if patterns:
            logger.runtime.info(
                f"[PII Replacement] Standalone column {col!r} (entity={entity}, patterns={patterns}){extra}"
            )
        else:
            logger.runtime.info(f"[PII Replacement] Standalone column {col!r} (entity={entity}){extra}")

    def _gather(col: str) -> ColumnEvidence:
        series = df_subset[col]
        name_label, demo_label = match_column_header(
            col, ENTITY_NAME_PATTERNS, DEMO_LABEL_PATTERNS, cfg.name_fuzzy_threshold
        )
        phone_min = 7 if name_label == "phone_number" else 10
        analysis = analyze_column_patterns(series, cfg, phone_min_digits=phone_min)
        value_entity = analysis["entity"] if analysis["structured"] else None
        # Never assign *replaceable* entities from values alone. Temporals
        # (identify-not-replaced) keep value evidence without a name match.
        if value_entity is not None and not name_supports_value_entity(name_label, value_entity):
            value_entity = None
        return ColumnEvidence(col, series, name_label, value_entity, analysis, demo_label)

    for col in df_subset.columns:
        ev = _gather(col)

        # 1) Numeric probes (dtype-driven; still require a supportive header).
        numeric_probe = probe_numeric_column(ev.series, ev.name_label)
        if numeric_probe == "date_of_birth":
            _add_standalone(
                col,
                "date_of_birth",
                ["%Y%m%d"],
                note="perturbed per record/group (not persona-tied)",
            )
            continue
        if numeric_probe in {"unique_identifier", "national_id", "ssn"}:
            # Sequential-integer skip is a unique_identifier heuristic (surrogate
            # keys). Numeric ssn / national_id columns must still be replaced.
            if numeric_probe == "unique_identifier" and looks_like_sequential_integer_id(ev.series):
                logger.user.warning(
                    f"[PII Replacement] Column {col!r} looks like a sequential integer id "
                    "(1, 2, 3, …); skipped — not treated as a unique identifier."
                )
                continue
            _add_standalone(col, numeric_probe, [])
            continue

        # 2) Name-matched entity: resolve EntitySpec → gates → allocate.
        if ev.name_label is not None:
            entity_spec = spec(ev.name_label)
            apply_path = effective_apply_path(ev.name_label, backend)
            if apply_path == "identify_only":
                identified_not_replaced.append(col)
                consumed.add(col)
                logger.runtime.info(
                    f"[PII Replacement] Identified column {col!r} (entity={ev.name_label}) — "
                    "excluded from replacement plan"
                )
                continue
            if entity_spec is not None and apply_path in {"persona", "standalone_map"}:
                skip = skip_reason_named_column(entity_spec, ev.series, ev.value_entity, apply_path)
                if skip is not None:
                    # Prefer the historical phrasing for value-match skips.
                    if entity_spec.requires_value_match or ev.name_label == "date_of_birth":
                        logger.user.warning(
                            f"[PII Replacement] Column {col!r} looks like {ev.name_label} by name but {skip}; skipped."
                        )
                    elif ev.name_label == "unique_identifier":
                        logger.user.warning(
                            f"[PII Replacement] Column {col!r} looks like a sequential integer id "
                            "(1, 2, 3, …); skipped — not treated as a unique identifier."
                        )
                    elif ev.name_label == "api_key":
                        logger.user.warning(
                            f"[PII Replacement] Column {col!r} looks like api_key by name/values but {skip}; skipped."
                        )
                    else:
                        logger.user.warning(f"[PII Replacement] Column {col!r} {skip}; skipped.")
                    continue

                if apply_path == "standalone_map":
                    patterns: list[str] = []
                    note = ""
                    if ev.name_label == "date_of_birth":
                        patterns = date_patterns(ev.series.dropna())
                        note = "perturbed per record/group (not persona-tied)"
                    _add_standalone(col, ev.name_label, patterns, note=note)
                    continue

                # Persona path: name agreement + column-derived role pools.
                role = _persona_role_key(col)
                had_label = any(ev.name_label in fields_by_persona.get(pid, {}) for pid in role_personas.get(role, []))
                persona = _allocate_persona(
                    col,
                    ev.name_label,
                    warn_collision=had_label,
                    require_name_agreement=True,
                )
                fields_by_persona.setdefault(persona, {})[ev.name_label] = col
                patterns = value_patterns(ev.series.dropna(), cfg) if ev.name_label == "phone_number" else []
                if patterns:
                    field_patterns_by_persona.setdefault(persona, {})[ev.name_label] = patterns
                    logger.runtime.info(
                        f"[PII Replacement] Persona-backed column {col!r} (entity={ev.name_label}, patterns={patterns})"
                    )
                else:
                    logger.runtime.info(
                        f"[PII Replacement] Persona-backed column {col!r} (entity={ev.name_label}, persona={persona})"
                    )
                consumed.add(col)
                continue

        # 3) Demographic matchers (read-only; may share a persona with later fields).
        if ev.demo_label:
            persona = _allocate_demo(col, ev.demo_label)
            demo_by_persona.setdefault(persona, {})[ev.demo_label] = col

        # 4) Identify-only temporals (value evidence, no replaceable name match).
        if (
            ev.value_entity is not None
            and is_identify_only(ev.value_entity)
            and ev.name_label != "date_of_birth"
            and ev.demo_label is None
        ):
            identified_not_replaced.append(col)
            consumed.add(col)
            logger.runtime.info(
                f"[PII Replacement] Identified temporal column {col!r} (entity={ev.value_entity}, "
                f"pattern={ev.analysis['pattern']}, coverage={ev.analysis['coverage']}) — "
                "excluded from replacement plan"
            )

    personas: list[dict[str, object]] = []
    for persona in sorted(set(fields_by_persona) | set(demo_by_persona), key=lambda p: (len(p), p)):
        fields = fields_by_persona.get(persona, {})
        demo = demo_by_persona.get(persona, {})
        if not fields and not demo:
            continue
        patterns_by_label = field_patterns_by_persona.get(persona, {})
        personas.append(
            {
                "persona": persona,
                "fields": {
                    label: {"column": col, "patterns": list(patterns_by_label.get(label) or [])}
                    for label, col in fields.items()
                },
                "match_persona_by": [
                    {"persona_attribute": attr, "column_name": demo[attr]}
                    for attr in demo_keys_for_backend(cfg.persona_backend)
                    if demo.get(attr)
                ],
            }
        )

    exclude = set(consumed)
    for persona_set in personas:
        matchers = cast(list[dict[str, str]], persona_set.get("match_persona_by") or [])
        exclude |= {entry["column_name"] for entry in matchers}
    exclude |= set(identified_not_replaced)
    free_text = detect_free_text_columns(df_subset, stats, exclude, cfg)
    return {
        "personas": personas,
        "free_text_columns": free_text,
        "standalone_columns": standalone,
        "identified_not_replaced": identified_not_replaced,
    }

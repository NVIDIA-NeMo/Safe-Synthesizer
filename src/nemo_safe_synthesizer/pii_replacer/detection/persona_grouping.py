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
from ..log_context import column_log_label
from ..models import (
    ColumnEvidence,
    DetectedPersona,
    DetectedPersonaField,
    DetectedStandalone,
    DiscoveryResult,
    PersonaMatchBy,
    StructuralGrain,
)
from ..patterns import date_patterns, split_full_name, value_patterns
from .column_names import (
    match_column_header,
    name_supports_value_entity,
    normalize_column_name_for_match,
)
from .value_recognizers import (
    analyze_column_patterns,
    looks_like_sequential_integer_id,
    probe_numeric_column,
)

logger = get_logger(__name__)

# Only these name parts participate in cross-column agreement checks.
_NAME_AGREE_LABELS = frozenset({"first_name", "last_name", "full_name"})
_NAME_AGREE_THRESHOLD = 0.85


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
    When no row is comparable (missing overlap), returns ``True`` so columns still
    merge; only a clear disagreement rate below the threshold forces a split.
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
    if comparable == 0:
        return True
    return (agree / comparable) >= _NAME_AGREE_THRESHOLD


def detect_structured_columns(df_subset: pd.DataFrame, stats: dict, cfg: Config) -> DiscoveryResult:
    """Detect persona-backed and standalone PII columns over a column subset.

    Per column: gather name/value evidence, resolve ``EntitySpec`` and apply path,
    run content gates, then allocate to persona, standalone, or identify-only.
    Persona membership respects structural ``grain`` from ``stats`` so a
    group-constant name and a record-varying email do not share one identity.

    Args:
        df_subset: Dataframe slice whose columns are classified.
        stats: Per-column stats from ``scoped_column_stats`` or ``column_stats``
            (expects a ``grain`` tag when a training group key was used).
        cfg: Engine configuration controlling thresholds and persona backend.

    Returns:
        Typed discovery result. ``free_text_columns`` is always empty here;
        discovery fills it via ``select_free_text_columns``.
    """
    backend = cfg.persona_backend
    fields_by_persona: dict[str, dict[str, str]] = {}
    field_patterns_by_persona: dict[str, dict[str, list[str]]] = {}
    demo_by_persona: dict[str, dict[str, str]] = {}
    grain_by_persona: dict[str, StructuralGrain] = {}
    role_personas: dict[str, list[str]] = {}
    empty_persona_seq = 0
    role_instance_count: dict[str, int] = {}
    standalone: list[DetectedStandalone] = []
    identified_not_replaced: list[str] = []
    consumed: set[str] = set()

    def _column_grain(col: str) -> StructuralGrain:
        raw = (stats.get(col) or {}).get("grain", "record")
        return cast(StructuralGrain, raw if raw in {"key", "group", "record"} else "record")

    def _persona_grain(grain: StructuralGrain) -> StructuralGrain:
        # Group-key columns are not persona fields; treat as group-constant.
        return "group" if grain == "key" else grain

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
        grain: StructuralGrain = "record",
    ) -> str:
        role = _persona_role_key(col)
        pool = role_personas.setdefault(role, [])
        persona_grain = _persona_grain(grain)
        disagreement: str | None = None
        for pid in pool:
            if grain_by_persona.get(pid) not in (None, persona_grain):
                continue
            if label in fields_by_persona.get(pid, {}) or label in demo_by_persona.get(pid, {}):
                continue
            if require_name_agreement and label in _NAME_AGREE_LABELS:
                existing = fields_by_persona.get(pid, {})
                if any(lbl in _NAME_AGREE_LABELS for lbl in existing) and not _persona_names_agree(
                    df_subset, existing, col, label
                ):
                    disagreement = pid
                    continue
            grain_by_persona.setdefault(pid, persona_grain)
            return pid
        if disagreement is not None:
            logger.user.warning(
                f"[PII Replacement] Column {column_log_label(col)!r} ({label}) does not agree with name columns on "
                f"persona {disagreement!r}; assigning a new persona (review the plan)."
            )
        elif warn_collision and pool:
            prior = pool[-1]
            prior_col = fields_by_persona.get(prior, {}).get(label) or demo_by_persona.get(prior, {}).get(label)
            prior_label = column_log_label(prior_col) if prior_col else prior_col
            logger.user.warning(
                f"[PII Replacement] Column {column_log_label(col)!r} shares entity {label!r} with {prior_label!r}; "
                f"assigning a new persona so both are replaced (review the plan)."
            )
        pid = _mint_persona(role)
        pool.append(pid)
        grain_by_persona[pid] = persona_grain
        return pid

    def _allocate_demo(col: str, label: str, grain: StructuralGrain) -> str:
        """Attach an unprefixed demographic to the earliest compatible persona.

        Prefixed demos (``patient_sex``) stay in their role pool. Bare ``sex`` /
        ``race`` columns are dataset-level attributes and join the first persona
        that does not already have that demographic, so they can condition the
        subject persona (e.g. ``patient``) even when role keys differ.
        """
        role = _persona_role_key(col)
        persona_grain = _persona_grain(grain)
        if role:
            return _allocate_persona(col, label, warn_collision=False, grain=grain)
        for pool in role_personas.values():
            for pid in pool:
                if grain_by_persona.get(pid) not in (None, persona_grain):
                    continue
                if label not in demo_by_persona.get(pid, {}) and label not in fields_by_persona.get(pid, {}):
                    grain_by_persona.setdefault(pid, persona_grain)
                    return pid
        return _allocate_persona(col, label, warn_collision=False, grain=grain)

    def _add_standalone(col: str, entity: str, patterns: list[str], *, note: str = "") -> None:
        standalone.append(DetectedStandalone(column=col, entity=entity, patterns=patterns))
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
        return ColumnEvidence(col, series, name_label, value_entity, analysis, demo_label, grain=_column_grain(col))

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
                    f"[PII Replacement] Column {column_log_label(col)!r} looks like a sequential integer id "
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
                # Route content gates through EntityHandler (same function DefaultHandler
                # wraps); keep the import local to avoid a cycle with entity_handlers.
                from ..entity_handlers import get_handler

                skip = get_handler(ev.name_label).skip_reason(
                    ev.series, ev.value_entity, apply_path, column_name=col, cfg=cfg
                )
                if skip is not None:
                    # Prefer the historical phrasing for value-match skips.
                    if entity_spec.requires_value_match or ev.name_label == "date_of_birth":
                        logger.user.warning(
                            f"[PII Replacement] Column {column_log_label(col)!r} looks like {ev.name_label} by name but {skip}; skipped."
                        )
                    elif ev.name_label == "unique_identifier" and "sequential" in skip:
                        logger.user.warning(
                            f"[PII Replacement] Column {column_log_label(col)!r} looks like a sequential integer id "
                            "(1, 2, 3, …); skipped — not treated as a unique identifier."
                        )
                    elif ev.name_label == "unique_identifier":
                        logger.user.warning(
                            f"[PII Replacement] Column {column_log_label(col)!r} looks like unique_identifier by name but "
                            f"{skip}; skipped."
                        )
                    elif ev.name_label == "api_key":
                        logger.user.warning(
                            f"[PII Replacement] Column {column_log_label(col)!r} looks like api_key by name/values but {skip}; skipped."
                        )
                    else:
                        logger.user.warning(
                            f"[PII Replacement] Column {column_log_label(col)!r} {skip}; skipped."
                        )
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
                    grain=ev.grain,
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
            persona = _allocate_demo(col, ev.demo_label, ev.grain)
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

    personas: list[DetectedPersona] = []
    for persona in sorted(set(fields_by_persona) | set(demo_by_persona), key=lambda p: (len(p), p)):
        fields = fields_by_persona.get(persona, {})
        demo = demo_by_persona.get(persona, {})
        if not fields and not demo:
            continue
        patterns_by_label = field_patterns_by_persona.get(persona, {})
        personas.append(
            DetectedPersona(
                persona=persona,
                fields={
                    label: DetectedPersonaField(column=col, patterns=list(patterns_by_label.get(label) or []))
                    for label, col in fields.items()
                },
                match_persona_by=[
                    PersonaMatchBy(persona_attribute=attr, column_name=demo[attr])
                    for attr in demo_keys_for_backend(cfg.persona_backend)
                    if demo.get(attr)
                ],
            )
        )

    # Free-text eligibility for the plan is decided later in discovery via
    # ``select_free_text_columns`` (NSS field types + structured-gate).
    return DiscoveryResult(
        personas=personas,
        free_text_columns=[],
        standalone_columns=standalone,
        identified_not_replaced=identified_not_replaced,
    )
